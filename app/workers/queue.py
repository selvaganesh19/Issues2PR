"""Durable job queue for the webhook-driven async flow.

Two interchangeable implementations share one interface:

* :class:`RedisQueue` — backed by Redis. Uses a Redis SET as an idempotency
  marker (keyed by GitHub delivery id), a list as the ready queue, a sorted set
  for delayed retries (exponential backoff), and a list as the dead-letter
  queue after ``max_retries`` attempts.
* :class:`InMemoryQueue` — a pure-Python fallback with identical semantics,
  used automatically when Redis is unavailable so imports and tests work
  without any infrastructure.

Both subclass :class:`BaseQueue`, which owns the retry / dead-letter policy and
delegates storage primitives to the subclass.

Use :func:`get_queue` to obtain the best available queue for the current
environment.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

# Defaults for the retry policy. Overridable via constructor arguments.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_BACKOFF_S = 5.0

# Redis key suffixes (namespaced under a common prefix).
_READY_SUFFIX = "ready"
_DELAYED_SUFFIX = "delayed"
_DEAD_SUFFIX = "dead"
_SEEN_SUFFIX = "seen"


@dataclass
class Job:
    """A unit of work flowing through the queue.

    Attributes:
        delivery_id: Stable, unique id used for idempotency (e.g. the GitHub
            ``X-GitHub-Delivery`` header). Duplicate enqueues are dropped.
        payload: Arbitrary JSON-serialisable job data (the webhook body).
        attempts: Number of processing attempts made so far.
        enqueued_at: Unix timestamp of first enqueue.
    """

    delivery_id: str
    payload: dict
    attempts: int = 0
    enqueued_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialise this job to a JSON string for storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> Job:
        """Deserialise a job previously produced by :meth:`to_json`."""
        raw = json.loads(data)
        return cls(
            delivery_id=str(raw["delivery_id"]),
            payload=dict(raw.get("payload", {})),
            attempts=int(raw.get("attempts", 0)),
            enqueued_at=float(raw.get("enqueued_at", time.time())),
        )


class BaseQueue:
    """Shared retry / dead-letter policy for the queue implementations.

    Subclasses implement the storage primitives (``_mark_seen``, ``_push``,
    ``_pop``, ``_schedule``, ``_promote_due``, ``_deadletter``,
    ``dead_letter_jobs``). This base owns:

    * idempotent :meth:`enqueue` (via ``_mark_seen``),
    * :meth:`dequeue` (promotes due delayed jobs first, then pops),
    * :meth:`handle_failure` (exponential backoff, then dead-letter).
    """

    def __init__(
        self,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_backoff_s: float = _DEFAULT_BASE_BACKOFF_S,
    ) -> None:
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

    # --- storage primitives (implemented by subclasses) ------------------
    def _mark_seen(self, delivery_id: str) -> bool:
        """Record ``delivery_id``; return True only the first time it is seen."""
        raise NotImplementedError

    def _push(self, job: Job) -> None:
        """Append ``job`` to the ready queue."""
        raise NotImplementedError

    def _pop(self, timeout: float) -> Job | None:
        """Pop the next ready job, blocking up to ``timeout`` seconds."""
        raise NotImplementedError

    def _schedule(self, job: Job, ready_at: float) -> None:
        """Store ``job`` for delayed re-delivery at ``ready_at`` (unix ts)."""
        raise NotImplementedError

    def _promote_due(self, now: float) -> None:
        """Move any delayed jobs whose ``ready_at <= now`` into the ready queue."""
        raise NotImplementedError

    def _deadletter(self, job: Job, error: str) -> None:
        """Move ``job`` to the dead-letter queue with a failure ``error``."""
        raise NotImplementedError

    def dead_letter_jobs(self) -> list[Job]:
        """Return all jobs currently in the dead-letter queue."""
        raise NotImplementedError

    # --- public API ------------------------------------------------------
    def enqueue(self, job: Job) -> bool:
        """Enqueue ``job``, idempotently by ``delivery_id``.

        Returns:
            True if the job was newly enqueued, False if a job with the same
            ``delivery_id`` was already seen (duplicate delivery -> dropped).
        """
        if not self._mark_seen(job.delivery_id):
            return False
        self._push(job)
        return True

    def dequeue(self, timeout: float = 5.0) -> Job | None:
        """Return the next job to process, or None if none is ready in time.

        Delayed retries whose backoff has elapsed are promoted to the ready
        queue before popping.
        """
        self._promote_due(time.time())
        return self._pop(timeout)

    def _backoff_delay(self, attempts: int) -> float:
        """Exponential backoff for the given (post-increment) attempt count."""
        return self.base_backoff_s * (2 ** max(0, attempts - 1))

    def handle_failure(self, job: Job, error: str) -> str:
        """Record a failed attempt and decide the job's fate.

        Increments ``attempts``; schedules a delayed retry with exponential
        backoff while attempts remain, otherwise dead-letters the job.

        Returns:
            ``"retry"`` if the job was rescheduled, ``"dead_letter"`` if it was
            moved to the dead-letter queue.
        """
        job.attempts += 1
        if job.attempts > self.max_retries:
            self._deadletter(job, error)
            return "dead_letter"
        self._schedule(job, time.time() + self._backoff_delay(job.attempts))
        return "retry"


class InMemoryQueue(BaseQueue):
    """Thread-safe, process-local queue. Used when Redis is unavailable.

    All state lives in memory and is lost on restart — adequate for tests and
    single-process local runs, but not durable across processes.
    """

    def __init__(
        self,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_backoff_s: float = _DEFAULT_BASE_BACKOFF_S,
    ) -> None:
        super().__init__(max_retries, base_backoff_s)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._ready: deque[Job] = deque()
        self._delayed: list[tuple[float, Job]] = []
        self._dead: list[Job] = []
        self._seen: set[str] = set()

    def _mark_seen(self, delivery_id: str) -> bool:
        with self._lock:
            if delivery_id in self._seen:
                return False
            self._seen.add(delivery_id)
            return True

    def _push(self, job: Job) -> None:
        with self._not_empty:
            self._ready.append(job)
            self._not_empty.notify()

    def _pop(self, timeout: float) -> Job | None:
        deadline = time.time() + timeout
        with self._not_empty:
            while not self._ready:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._not_empty.wait(remaining)
            return self._ready.popleft()

    def _schedule(self, job: Job, ready_at: float) -> None:
        with self._lock:
            self._delayed.append((ready_at, job))

    def _promote_due(self, now: float) -> None:
        with self._not_empty:
            still_waiting: list[tuple[float, Job]] = []
            promoted = False
            for ready_at, job in self._delayed:
                if ready_at <= now:
                    self._ready.append(job)
                    promoted = True
                else:
                    still_waiting.append((ready_at, job))
            self._delayed = still_waiting
            if promoted:
                self._not_empty.notify_all()

    def _deadletter(self, job: Job, error: str) -> None:
        with self._lock:
            self._dead.append(job)

    def dead_letter_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._dead)


class RedisQueue(BaseQueue):
    """Durable, cross-process queue backed by Redis.

    Keys (all prefixed with ``key_prefix``):
      * ``<prefix>:ready``   — list; the ready queue (``LPUSH`` / ``BRPOP``).
      * ``<prefix>:delayed`` — sorted set; delayed retries scored by ready-at ts.
      * ``<prefix>:dead``    — list; dead-lettered jobs.
      * ``<prefix>:seen``    — set; idempotency markers by ``delivery_id``.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "issue2pr:jobs",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_backoff_s: float = _DEFAULT_BASE_BACKOFF_S,
    ) -> None:
        super().__init__(max_retries, base_backoff_s)

        if redis_url.startswith("fake://"):
            # In-process Redis emulation: real RedisQueue code path, no server,
            # no Docker. Great for local dev and integration tests on Windows.
            import fakeredis

            self._redis = fakeredis.FakeRedis(decode_responses=True)
        else:
            import redis  # imported lazily so the module loads without redis

            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        # Fail fast if the server is not actually reachable.
        self._redis.ping()
        self._prefix = key_prefix
        self._ready_key = f"{key_prefix}:{_READY_SUFFIX}"
        self._delayed_key = f"{key_prefix}:{_DELAYED_SUFFIX}"
        self._dead_key = f"{key_prefix}:{_DEAD_SUFFIX}"
        self._seen_key = f"{key_prefix}:{_SEEN_SUFFIX}"

    def _mark_seen(self, delivery_id: str) -> bool:
        # SADD returns 1 when the member is new, 0 when it already existed.
        return bool(self._redis.sadd(self._seen_key, delivery_id))

    def _push(self, job: Job) -> None:
        self._redis.lpush(self._ready_key, job.to_json())

    def _pop(self, timeout: float) -> Job | None:
        # BRPOP blocks up to `timeout` seconds; timeout=0 would block forever,
        # so clamp to at least 1 second of integer granularity.
        popped = self._redis.brpop([self._ready_key], timeout=max(1, int(timeout)))
        if popped is None:
            return None
        _key, data = popped
        return Job.from_json(data)

    def _schedule(self, job: Job, ready_at: float) -> None:
        self._redis.zadd(self._delayed_key, {job.to_json(): ready_at})

    def _promote_due(self, now: float) -> None:
        # Fetch all delayed jobs whose ready-at has elapsed, then move each into
        # the ready list and remove it from the delayed set.
        due = self._redis.zrangebyscore(self._delayed_key, min="-inf", max=now)
        for data in due:
            # Only push if we win the race to remove it (avoids double-delivery
            # across concurrent workers).
            if self._redis.zrem(self._delayed_key, data):
                self._redis.lpush(self._ready_key, data)

    def _deadletter(self, job: Job, error: str) -> None:
        record = {
            "job": json.loads(job.to_json()),
            "error": error,
            "failed_at": time.time(),
        }
        self._redis.lpush(self._dead_key, json.dumps(record))

    def dead_letter_jobs(self) -> list[Job]:
        records = self._redis.lrange(self._dead_key, 0, -1)
        jobs: list[Job] = []
        for record in records:
            raw = json.loads(record)
            jobs.append(Job.from_json(json.dumps(raw["job"])))
        return jobs


def get_queue(settings: Settings | None = None, **kwargs) -> BaseQueue:
    """Return a :class:`RedisQueue` if Redis is reachable, else :class:`InMemoryQueue`.

    Args:
        settings: Optional :class:`~app.config.Settings`; loaded via
            ``get_settings()`` when omitted. Its ``redis_url`` selects the
            Redis server.
        **kwargs: Forwarded to the queue constructor (e.g. ``max_retries``,
            ``base_backoff_s``, ``key_prefix``).

    The Redis import and connection are attempted defensively; any failure
    (package missing, server down) transparently falls back to the in-memory
    implementation so local/test usage needs no infrastructure.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    prefix = kwargs.pop("key_prefix", "issue2pr:jobs")
    try:
        return RedisQueue(settings.redis_url, key_prefix=prefix, **kwargs)
    except Exception:  # noqa: BLE001 - any failure -> in-memory fallback
        return InMemoryQueue(**kwargs)


def run_worker(
    queue: BaseQueue,
    handler: Callable[[Job], object],
    *,
    stop: threading.Event | None = None,
    poll_timeout: float = 5.0,
) -> None:
    """Consume jobs from ``queue`` and process each with ``handler``.

    Loops until ``stop`` is set (or forever if not provided). Each dequeued job
    is passed to ``handler``; any exception routes the job through
    :meth:`BaseQueue.handle_failure` (retry with backoff, then dead-letter).

    Args:
        queue: The queue to consume from.
        handler: Callable invoked with each :class:`Job`. Raising signals
            failure.
        stop: Optional event; when set, the loop exits after the current poll.
        poll_timeout: Blocking dequeue timeout, in seconds.
    """
    while stop is None or not stop.is_set():
        job = queue.dequeue(timeout=poll_timeout)
        if job is None:
            continue
        try:
            handler(job)
        except Exception as exc:  # noqa: BLE001 - failure is expected/handled
            queue.handle_failure(job, str(exc))
