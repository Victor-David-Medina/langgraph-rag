"""Tests for the retrieve -> grade -> generate loop.

Run: python -m pytest tests/test_graph.py

Drafted 2026-11-23 against graph.py / corpus.py / embeddings.py as read on
main that day. All tests use the labeled hashed bag-of-words fallback
embedder, so the suite is deterministic and runs fully offline (no model
download). The grade test uses synthetic documents so it verifies the
threshold logic itself, not embedder semantics.
"""
import numpy as np
import pytest

from corpus import DOCS
from embeddings import HashedEmbedder, cosine
from graph import (
    GRADE_THRESHOLD,
    MAX_RETRIES,
    MIN_GRADED,
    grade,
    rewrite_query,
    route_after_grade,
)


@pytest.fixture(scope="module")
def embedder():
    return HashedEmbedder()


def _state(**over):
    base = {
        "question": "",
        "active_query": "",
        "retrieved": [],
        "graded": [],
        "retries": 0,
        "answer": "",
        "trace": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- corpus

def test_corpus_has_seven_docs_with_unique_ids():
    assert len(DOCS) == 7
    ids = [d["id"] for d in DOCS]
    assert len(set(ids)) == 7
    for d in DOCS:
        assert d["title"] and d["text"]


# ---------------------------------------------------------------- embeddings

def test_cosine_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(0.0)


def test_fallback_embedder_is_deterministic(embedder):
    a = embedder.encode(["routing model calls through a gateway"])[0]
    b = embedder.encode(["routing model calls through a gateway"])[0]
    assert np.allclose(a, b)


# ---------------------------------------------------------------- rewrite node

def test_rewrite_expands_known_abbreviation():
    out = rewrite_query(_state(active_query="What is RAG?"))
    assert "retrieval augmented generation" in out["active_query"]
    assert out["retries"] == 1


def test_rewrite_without_abbreviation_keeps_query_text():
    q = "How do I bake sourdough bread at home?"
    out = rewrite_query(_state(active_query=q))
    assert out["active_query"] == q
    assert out["retries"] == 1


# ---------------------------------------------------------------- grade node

def test_grade_keeps_above_threshold_and_drops_below(embedder):
    question = "alpha beta gamma delta"
    on_topic = "alpha beta gamma delta epsilon zeta"
    off_topic = "one two three four five six"
    state = _state(
        question=question,
        retrieved=[
            ("on-topic", "On topic", on_topic, 0.99),
            ("off-topic", "Off topic", off_topic, 0.99),
        ],
    )
    out = grade(state, embedder)
    kept = {doc_id for doc_id, *_ in out["graded"]}
    assert "on-topic" in kept
    assert "off-topic" not in kept
    assert all(g >= GRADE_THRESHOLD for *_, g in out["graded"])


# ---------------------------------------------------------------- routing / termination

def test_route_generates_when_enough_graded():
    state = _state(graded=[("a", "t", "x", 0.9)] * MIN_GRADED, retries=0)
    assert route_after_grade(state) == "generate"


def test_route_rewrites_when_too_few_graded_and_retries_left():
    state = _state(graded=[], retries=0)
    assert route_after_grade(state) == "rewrite"


def test_route_terminates_at_max_retries():
    # The bound that guarantees the graph always terminates: once retries
    # are exhausted, routing goes to generate (best effort), never rewrite.
    state = _state(graded=[], retries=MAX_RETRIES)
    assert route_after_grade(state) == "generate"


def test_rewrite_increments_retries_so_the_loop_is_bounded():
    out = rewrite_query(_state(active_query="q", retries=MAX_RETRIES - 1))
    assert out["retries"] == MAX_RETRIES
    merged = _state(graded=[], retries=out["retries"])
    assert route_after_grade(merged) == "generate"
