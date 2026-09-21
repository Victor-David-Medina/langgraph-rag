"""Embedding layer.

Preferred path: real dense embeddings from sentence-transformers
(all-MiniLM-L6-v2, 384 dims, ~90MB download, runs on CPU).

Fallback: a clearly labeled hashed bag-of-words embedder implemented in pure
numpy. It is deterministic and has zero dependencies, but it is keyword
matching, not semantic similarity, so unrelated phrasings will not match.
The fallback exists so the graph still runs in an offline environment;
it is not presented as equivalent to real embeddings.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

REAL_MODEL_NAME = "all-MiniLM-L6-v2"
REAL_DIM = 384
FALLBACK_DIM = 256


class RealEmbedder:
    name = f"sentence-transformers:{REAL_MODEL_NAME}"
    dim = REAL_DIM

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(REAL_MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class HashedEmbedder:
    """CLEARLY-LABELED FALLBACK: hashed bag-of-words, not semantic embeddings."""

    name = "fallback:hashed-bow-256"
    dim = FALLBACK_DIM

    def __init__(self) -> None:
        print("WARNING: using hashed bag-of-words fallback, NOT real embeddings.")

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def get_embedder():
    """Return (embedder, used_fallback). Prefer real embeddings."""
    try:
        return RealEmbedder(), False
    except Exception as exc:  # missing package, no network, etc.
        print(f"Real embedder unavailable ({exc}); falling back.")
        return HashedEmbedder(), True


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)
