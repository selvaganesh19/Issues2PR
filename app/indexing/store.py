"""FAISS-backed vector store for semantic code search.

:class:`FaissStore` builds a similarity index over :class:`~app.indexing.chunker.Chunk`
objects, persists it to a ``.faiss`` directory (index file + JSON metadata), and
answers nearest-neighbour queries. It is the backing store used by
:func:`app.tools.search.semantic_search` when a :class:`ToolContext` has ``index``
set.

Optional-dependency handling
----------------------------
``faiss`` and ``numpy`` are part of the optional ``index`` extra
(``pip install issue2pr[index]``). They are imported lazily inside the helpers
below so that ``import app.indexing.store`` succeeds even when the extra is not
installed; a helpful :class:`IndexUnavailableError` is raised only when index
functionality is actually exercised.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

from app.indexing.chunker import Chunk

if TYPE_CHECKING:
    from app.indexing.embedder import Embedder

# Filenames used inside the persisted ``.faiss`` directory.
_INDEX_FILE = "index.faiss"
_META_FILE = "meta.json"

DEFAULT_TOP_K = 5


class IndexUnavailableError(RuntimeError):
    """Raised when faiss/numpy are required but not installed."""


def _import_numpy() -> Any:
    """Import numpy lazily, raising a helpful error if it is missing."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise IndexUnavailableError(
            "numpy is required for the semantic index; install with "
            "'pip install issue2pr[index]'"
        ) from exc
    return np


def _import_faiss() -> Any:
    """Import faiss lazily, raising a helpful error if it is missing."""
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise IndexUnavailableError(
            "faiss-cpu is required for the semantic index; install with "
            "'pip install issue2pr[index]'"
        ) from exc
    return faiss


def _normalize(matrix: Any) -> Any:
    """L2-normalize rows so inner product equals cosine similarity."""
    np = _import_numpy()
    arr = np.asarray(matrix, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class FaissStore:
    """A persisted FAISS index over source chunks.

    Similarity uses cosine distance implemented as inner product over
    L2-normalized vectors (``IndexFlatIP``). The store keeps the chunk metadata
    alongside the vectors so query results can be returned as
    :class:`~app.indexing.chunker.Chunk` objects.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        """Create an (empty) store.

        Args:
            embedder: Used to embed chunk texts on :meth:`build` and query
                strings on :meth:`query`. Required for those operations but not
                for :meth:`load` of an already-built index that is never queried.
        """
        self.embedder = embedder
        self._index: Any = None
        self._chunks: list[Chunk] = []
        self._dim: int | None = None

    # -- construction -----------------------------------------------------

    def build(self, chunks: list[Chunk]) -> FaissStore:
        """Embed ``chunks`` and build the in-memory FAISS index.

        Args:
            chunks: Chunks to index. An empty list produces an empty store.

        Returns:
            ``self`` for chaining.

        Raises:
            IndexUnavailableError: If faiss/numpy are not installed.
            ValueError: If no embedder was provided.
        """
        if self.embedder is None:
            raise ValueError("FaissStore.build requires an embedder")

        faiss = _import_faiss()
        self._chunks = list(chunks)
        if not self._chunks:
            self._index = None
            self._dim = None
            return self

        vectors = self.embedder.embed([c.text for c in self._chunks])
        matrix = _normalize(vectors)
        self._dim = int(matrix.shape[1])
        index = faiss.IndexFlatIP(self._dim)
        index.add(matrix)
        self._index = index
        return self

    # -- query ------------------------------------------------------------

    def query(self, text: str, k: int = DEFAULT_TOP_K) -> list[Chunk]:
        """Return the ``k`` chunks most similar to ``text``.

        Each returned chunk is a copy carrying its similarity ``score`` (cosine,
        higher is more similar). Returns an empty list for an empty index.

        Raises:
            IndexUnavailableError: If faiss/numpy are not installed.
            ValueError: If no embedder was provided.
        """
        if self._index is None or not self._chunks:
            return []
        if self.embedder is None:
            raise ValueError("FaissStore.query requires an embedder")

        query_vec = _normalize([self.embedder.embed_one(text)])
        top = min(k, len(self._chunks))
        scores, indices = self._index.search(query_vec, top)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            base = self._chunks[int(idx)]
            results.append(
                Chunk(
                    path=base.path,
                    start_line=base.start_line,
                    end_line=base.end_line,
                    text=base.text,
                    score=float(score),
                )
            )
        return results

    def search(self, text: str, k: int = DEFAULT_TOP_K) -> str:
        """Query the index and format results as a human-readable string.

        This is the entry point used by :func:`app.tools.search.semantic_search`.
        """
        chunks = self.query(text, k)
        if not chunks:
            return f"semantic_search: no results for {text!r}."
        blocks: list[str] = []
        for chunk in chunks:
            score = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
            snippet = chunk.text if len(chunk.text) <= 800 else chunk.text[:800] + "\n..."
            blocks.append(f"[{chunk.location()}] (score={score})\n{snippet}")
        return "\n\n".join(blocks)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Persist the index and chunk metadata to a ``.faiss`` directory.

        Args:
            path: Target directory (created if absent).

        Returns:
            The resolved directory path.

        Raises:
            IndexUnavailableError: If faiss is not installed.
            ValueError: If the store has not been built.
        """
        if self._index is None:
            raise ValueError("cannot save an empty/unbuilt index")
        faiss = _import_faiss()
        out_dir = pathlib.Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out_dir / _INDEX_FILE))
        meta = {
            "dim": self._dim,
            "model": getattr(self.embedder, "model", None),
            "chunks": [
                {
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                }
                for c in self._chunks
            ],
        }
        (out_dir / _META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        return out_dir.resolve()

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        embedder: Embedder | None = None,
    ) -> FaissStore:
        """Load a previously saved index from a ``.faiss`` directory.

        Args:
            path: Directory previously written by :meth:`save`.
            embedder: Embedder to attach for querying (required to call
                :meth:`query`/:meth:`search` afterwards).

        Returns:
            A populated :class:`FaissStore`.

        Raises:
            IndexUnavailableError: If faiss is not installed.
            FileNotFoundError: If the directory or its files are missing.
        """
        faiss = _import_faiss()
        in_dir = pathlib.Path(path)
        index_path = in_dir / _INDEX_FILE
        meta_path = in_dir / _META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"no saved index at {in_dir}")

        store = cls(embedder=embedder)
        store._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store._dim = meta.get("dim")
        store._chunks = [
            Chunk(
                path=item["path"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                text=item["text"],
            )
            for item in meta.get("chunks", [])
        ]
        return store

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
