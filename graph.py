"""LangGraph retrieve -> grade -> generate loop.

Nodes: retrieve, grade, rewrite_query, generate.
Edges: retrieve -> grade -> {generate | rewrite_query -> retrieve}.
The rewrite loop is bounded by MAX_RETRIES so the graph always terminates.
"""
from __future__ import annotations

import os
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from corpus import DOCS
from embeddings import cosine, get_embedder

COLLECTION = "skill_proof_docs"
TOP_K = 4
GRADE_THRESHOLD = 0.30
MIN_GRADED = 2
MAX_RETRIES = 2

# abbreviation expansions used by the rewrite_query node
EXPANSIONS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "mcp": "model context protocol",
    "slm": "small language model",
}


class RAGState(TypedDict):
    question: str
    active_query: str
    retrieved: list  # [(doc_id, title, text, search_score)]
    graded: list     # [(doc_id, title, text, grade)]
    retries: int
    answer: str
    trace: list      # human-readable node log for the run script


# ---------------------------------------------------------------- setup

def build_index(data_dir: str):
    """Embed the corpus and load it into Qdrant local mode. Returns (client, embedder)."""
    embedder, used_fallback = get_embedder()
    client = QdrantClient(path=data_dir)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    texts = [d["text"] for d in DOCS]
    vecs = embedder.encode(texts)
    points = [
        PointStruct(
            id=i,
            vector=vecs[i].tolist(),
            payload={"doc_id": d["id"], "title": d["title"], "text": d["text"]},
        )
        for i, d in enumerate(DOCS)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return client, embedder, used_fallback


# ---------------------------------------------------------------- nodes

def _log(state: RAGState, msg: str) -> None:
    state["trace"].append(msg)


def retrieve(state: RAGState, client: QdrantClient, embedder) -> dict:
    q = state["active_query"]
    qv = embedder.encode([q])[0]
    hits = client.query_points(collection_name=COLLECTION, query=qv.tolist(), limit=TOP_K).points
    retrieved = [
        (h.payload["doc_id"], h.payload["title"], h.payload["text"], round(h.score, 3))
        for h in hits
    ]
    _log(state, f"[retrieve] query={q!r} -> {len(retrieved)} hits: " +
         ", ".join(f"{doc_id} ({score})" for doc_id, _, _, score in retrieved))
    return {"retrieved": retrieved}


def grade(state: RAGState, embedder) -> dict:
    """Score each hit against the ORIGINAL question with cosine similarity; keep the good ones."""
    qv = embedder.encode([state["question"]])[0]
    graded = []
    for doc_id, title, text, search_score in state["retrieved"]:
        dv = embedder.encode([text])[0]
        g = round(cosine(qv, dv), 3)
        verdict = "keep" if g >= GRADE_THRESHOLD else "drop"
        _log(state, f"[grade] {doc_id}: similarity={g} -> {verdict} (threshold {GRADE_THRESHOLD})")
        if g >= GRADE_THRESHOLD:
            graded.append((doc_id, title, text, g))
    graded.sort(key=lambda t: t[3], reverse=True)
    return {"graded": graded}


def rewrite_query(state: RAGState) -> dict:
    q = state["active_query"]
    lowered = q.lower()
    additions = [exp for abbr, exp in EXPANSIONS.items()
                 if abbr in re.findall(r"[a-z]+", lowered)]
    new_q = q + (" " + " ".join(additions) if additions else "")
    retries = state["retries"] + 1
    _log(state, f"[rewrite] attempt {retries}: {q!r} -> {new_q!r}")
    return {"active_query": new_q, "retries": retries}


def _litellm_settings():
    """Read LiteLLM gateway config from the environment.

    LITELLM_BASE_URL: e.g. http://hermes-rig:4000/v1 (required to use the gateway)
    LITELLM_API_KEY:  proxy key (optional; many local proxies accept anything)
    LITELLM_MODEL:    model name as configured in the proxy (default gpt-4o-mini)
    Returns None when the gateway is not configured.
    """
    base_url = os.environ.get("LITELLM_BASE_URL", "").strip()
    if not base_url:
        return None
    return {
        "base_url": base_url,
        "api_key": os.environ.get("LITELLM_API_KEY", "").strip() or "not-needed",
        "model": os.environ.get("LITELLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
    }


def _gateway_answer(state: RAGState, settings: dict) -> str:
    """Generate with a real model through the LiteLLM gateway (OpenAI-compatible)."""
    from openai import OpenAI

    context = "\n\n".join(
        f"[{title}]\n{text}" for _, title, text, _ in state["graded"][:3]
    )
    client = OpenAI(base_url=settings["base_url"], api_key=settings["api_key"], timeout=30)
    resp = client.chat.completions.create(
        model=settings["model"],
        temperature=0.2,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the context below. "
                    "If the context does not contain the answer, say so plainly."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def _extractive_answer(state: RAGState) -> str:
    """Extractive fallback: highest-overlap sentences from graded docs.

    Used when no LiteLLM gateway is configured, or when the gateway call
    fails. Deliberately offline so the example runs with no API key.
    """
    q_words = set(re.findall(r"[a-z0-9]+", state["question"].lower()))
    sentences = []
    for doc_id, title, text, g in state["graded"][:3]:
        for s in re.split(r"(?<=[.!?])\s+", text):
            words = set(re.findall(r"[a-z0-9]+", s.lower()))
            overlap = len(q_words & words)
            sentences.append((overlap, g, s.strip(), title))
    sentences.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top = sentences[:3]
    lines = [f"Question: {state['question']}", ""]
    if not top or top[0][0] == 0:
        lines.append("I could not find relevant context in the corpus for this question.")
    else:
        lines.append("Answer (extracted from graded docs):")
        for _, _, s, title in top:
            lines.append(f"- {s} [{title}]")
    return "\n".join(lines)


def generate(state: RAGState) -> dict:
    """Answer from the graded docs.

    Preferred path: a real model call through the LiteLLM gateway, so the
    model (and provider) is a config change, not a code change. Falls back
    to the extractive answer when the gateway is unconfigured or errors.
    """
    settings = _litellm_settings()
    if settings and state["graded"]:
        try:
            answer = _gateway_answer(state, settings)
            _log(state, f"[generate] via LiteLLM gateway model={settings['model']} "
                        f"({len(state['graded'])} docs in context)")
            return {"answer": f"Question: {state['question']}\n\n{answer}"}
        except Exception as exc:
            _log(state, f"[generate] gateway failed ({exc}); extractive fallback")
    answer = _extractive_answer(state)
    _log(state, f"[generate] extractive fallback from {len(state['graded'])} graded docs")
    return {"answer": answer}


def route_after_grade(state: RAGState) -> str:
    if len(state["graded"]) >= MIN_GRADED:
        return "generate"
    if state["retries"] < MAX_RETRIES:
        return "rewrite"
    return "generate"  # best effort: answer with what we have


# ---------------------------------------------------------------- graph

def build_graph(client: QdrantClient, embedder):
    g = StateGraph(RAGState)
    g.add_node("retrieve", lambda s: retrieve(s, client, embedder))
    g.add_node("grade", lambda s: grade(s, embedder))
    g.add_node("rewrite", rewrite_query)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()
