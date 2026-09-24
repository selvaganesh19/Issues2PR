"""Tests for the GitHub webhook receiver (:mod:`app.api.webhooks`).

Covers HMAC signature verification (accept valid, reject tampered / missing)
and duplicate-delivery handling via the in-memory dedup set (no live Redis).
"""

from __future__ import annotations

import hashlib
import hmac

from app.api import webhooks
from app.api.webhooks import _mark_seen, verify_signature


def _sign(secret: str, body: bytes) -> str:
    """Produce a valid ``X-Hub-Signature-256`` header value for ``body``."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_verify_signature_accepts_valid():
    secret = "top-secret"
    body = b'{"action":"labeled"}'
    sig = _sign(secret, body)
    assert verify_signature(secret, body, sig) is True


def test_verify_signature_rejects_tampered_body():
    secret = "top-secret"
    body = b'{"action":"labeled"}'
    sig = _sign(secret, body)
    tampered = b'{"action":"deleted"}'
    assert verify_signature(secret, tampered, sig) is False


def test_verify_signature_rejects_tampered_signature():
    secret = "top-secret"
    body = b'{"action":"labeled"}'
    sig = _sign(secret, body)
    bad = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    assert verify_signature(secret, body, bad) is False


def test_verify_signature_rejects_missing_secret_or_header():
    body = b"payload"
    assert verify_signature(None, body, _sign("s", body)) is False
    assert verify_signature("s", body, None) is False


def test_duplicate_delivery_dedup():
    # Use a unique id so we don't collide with the module-level bounded set.
    delivery_id = "test-delivery-unique-0001"
    webhooks._SEEN_DELIVERIES.pop(delivery_id, None)

    assert _mark_seen(delivery_id) is False  # first time -> not previously seen
    assert _mark_seen(delivery_id) is True  # redelivery -> recognised as duplicate

    # Clean up so repeated test runs stay independent.
    webhooks._SEEN_DELIVERIES.pop(delivery_id, None)


def test_seen_set_is_bounded():
    # Filling past the cap must never grow the set without bound.
    cap = webhooks._SEEN_MAX
    for i in range(cap + 50):
        _mark_seen(f"bound-check-{i}")
    assert len(webhooks._SEEN_DELIVERIES) <= cap
