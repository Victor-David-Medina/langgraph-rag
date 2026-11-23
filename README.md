# LangGraph Retrieve-Grade-Generate RAG Loop

A small, working skills-proof project: a LangGraph agent that answers questions
from a 7-document local corpus using a **retrieve → grade → generate** loop,
with **Qdrant in local mode** as the vector database.

This is a portfolio example, not production. It runs fully offline with no API
key: the `generate` node is extractive (it quotes the best sentences from the
graded docs) so the loop can be demonstrated end to end without a model call.
In a real deployment that node calls an LLM, which is exactly where LiteLLM
slots in (see below).

**Results:** 11 tests passing (`tests/test_graph.py`, including the
termination bound) · full run transcripts committed at
[`examples/sample-run.md`](examples/sample-run.md): the on-corpus question
retrieves 4 docs, grades 1 above threshold, and answers from that doc; the
off-corpus question grades 0, retries twice, and abstains · runs fully
offline, no API key.

**Status:** complete, maintained best-effort.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
# or ask your own question:
.venv/bin/python run.py "What does a retrieval grader do?"
```

Expected: the script indexes the corpus into `./qdrant_data/`, runs the graph,
prints a node-by-node trace, then prints the final answer. The default sample
question is chosen to match the corpus, so grading passes on the first pass.
Try an unrelated question (e.g. `"How do I bake sourdough?"`) to watch the
rewrite-and-retry cycle fire before the graph answers best-effort.

## Graph shape

```
retrieve --> grade --(enough good docs)--> generate --> END
                |
                --(too few pass, retries left)--> rewrite --> retrieve
```

- **retrieve**: embeds the active query and pulls top-4 hits from Qdrant.
- **grade**: re-scores each hit against the original question with cosine
  similarity and keeps only hits above a threshold. Search score and grade are
  separate on purpose: the index ranks candidates, the grader judges evidence.
- **rewrite**: expands abbreviations (e.g. "RAG" → "retrieval augmented
  generation") and retries retrieval. Bounded by `MAX_RETRIES = 2`, so the
  graph always terminates.
- **generate**: composes the answer from the graded docs only.

## Why LangGraph instead of plain sequential chaining

A plain chain is `search()` then `answer()`: two function calls in a row, with
intermediate values trapped inside closures. That is fine until retrieval is
bad, and then you rerun everything blind.

LangGraph earns its place here in three specific ways:

1. **Explicit state.** The question, retrieved docs, grades, retry count, and
   trace all live in one `RAGState` object every node reads and writes. When
   the answer is wrong, you can see exactly which node produced which value
   instead of guessing.
2. **Cycles with a termination guarantee.** The grade → rewrite → retrieve loop
   is a real cycle in the graph, not a `while` loop bolted onto a chain. The
   retry counter in state makes the bound visible and checkable.
3. **Human-in-the-loop points.** Because state is explicit, you can pause
   before `generate`, show a person the graded docs, and let them approve,
   edit, or reject the evidence before any answer is composed. In a chain you
   would have to restructure the code to add that checkpoint; in LangGraph it
   is a node-level interrupt on state that already exists.

## Why Qdrant

- **Local mode needs nothing.** `QdrantClient(path=...)` embeds the storage
  engine, so there is no Docker container, no server process, and no network
  port. For a 7-document proof that is the difference between running and
  yak-shaving.
- **Same code scales up.** The client calls used here (`create_collection`,
  `upsert`, `query_points`) work unchanged against a hosted Qdrant cluster, so
  the example is not a throwaway API.
- **Payloads ride with vectors.** Each point stores the doc id plus its title and full text
  as payload, so retrieval returns usable context in one call instead of a
  second lookup.

Why not pgvector for this example: pgvector is the right call when you already
run Postgres and want vectors next to relational data, but it would add a
database server to a project that otherwise has zero infrastructure.

## Where LiteLLM fits (now wired)

David runs his own stack behind a LiteLLM proxy, so the model is never called
directly. In this graph, LiteLLM slots into exactly one place: **the
`generate` node, as the model-gateway layer**. This is now implemented, not
just documented.

The `generate()` node checks three environment variables:

| Variable | Meaning | Example |
|---|---|---|
| `LITELLM_BASE_URL` | Proxy endpoint (required to use the gateway) | `http://hermes-rig:4000/v1` |
| `LITELLM_API_KEY` | Proxy key (optional; many local proxies accept anything) | `sk-...` |
| `LITELLM_MODEL` | Model name as configured in the proxy (default `gpt-4o-mini`) | `ollama/llama3.1` |

Behavior:

- **Gateway set and reachable:** graded docs are injected as context and a
  real model writes the answer. Swapping providers is a proxy config change;
  the agent code never notices.
- **Gateway set but unreachable, or unset:** the node logs the failure and
  falls back to the extractive answer, so the example still runs fully
  offline with no API key.

The rest of the graph stays the same, because LiteLLM presents every provider (or a local Ollama
model) behind one request shape.

## What breaks at scale

Honest limits of this example, stated plainly:

- **The grader is cosine similarity, not judgment.** It measures vector
  closeness, which correlates with relevance but can be fooled by a chunk that
  uses the right words about the wrong thing. A real grader is a small model
  call with a rubric.
- **No chunking strategy.** Seven short docs fit in memory whole. Real corpora
  need a real chunking strategy, and retrieval quality lives or dies
  on those choices.
- **Flat threshold, no calibration.** `GRADE_THRESHOLD = 0.30` was picked by
  inspection on this corpus. Different embedding models and domains need
  different thresholds, ideally tuned on labeled examples.
- **Single-collection, single-tenant.** There is no namespacing, no access
  control, and no handling of stale or contradictory documents.
- **Extractive fallback.** Quoting sentences is honest for a demo but cannot
  synthesize, compare, or admit uncertainty the way a model call can. That is
  why the gateway path is preferred whenever `LITELLM_BASE_URL` is set.

## Files

- `run.py`: builds the index, runs the graph, prints trace + answer
- `graph.py`: LangGraph state, nodes (retrieve/grade/rewrite/generate), edges
- `corpus.py`: 7 short sample docs on AI infrastructure topics
- `embeddings.py`: sentence-transformers wrapper plus a labeled hashed
  bag-of-words fallback (used only if the real model cannot load)
- `requirements.txt`: langgraph, qdrant-client, sentence-transformers, numpy
- `tests/test_graph.py`: pytest suite over grading, query rewriting, and the
  routing bound that guarantees termination
- `examples/sample-run.md`: two full run transcripts (on-corpus and
  off-corpus), captured with the real embedder

## Sample trace

An off-corpus question, run with the offline hashed-bow fallback embedder (no model download). The grading node drops every hit below the similarity threshold, the rewrite loop retries twice, and the generator abstains instead of inventing an answer:

```
[retrieve] query='...' -> 4 hits: qdrant-local (0.274), mcp-basics (0.24), litellm-gateway (0.231), langgraph-state (0.182)
[grade] qdrant-local: similarity=0.274 -> drop (threshold 0.3)
[grade] mcp-basics: similarity=0.24 -> drop (threshold 0.3)
[grade] litellm-gateway: similarity=0.231 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.182 -> drop (threshold 0.3)
[rewrite] attempt 1: '...' -> '...'
[retrieve] query='...' -> 4 hits
[grade] ... -> drop (threshold 0.3)
[rewrite] attempt 2: '...' -> '...'
[generate] extractive fallback from 0 graded docs

FINAL ANSWER
I could not find relevant context in the corpus for this question.
```

Bounded retries, no silent failure, no invented answer.

Full transcripts of two runs with the real embedder (the default question
and the sourdough question, with measured retrieve/grade counts) are
committed at [`examples/sample-run.md`](examples/sample-run.md).
