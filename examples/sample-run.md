# Sample runs

Two real runs of `run.py`, captured 2026-11-23 with the full requirements
installed (embedder: `sentence-transformers:all-MiniLM-L6-v2`, Qdrant local
mode, no LiteLLM gateway configured, so `generate` uses the extractive
fallback). Output is verbatim except for the model-download progress bars,
which were removed. The local data directory in the first line differs by
machine.

Measured summary:

| Run | Retrieved | Graded above 0.30 | Rewrites | Outcome |
|---|---|---|---|---|
| Default question (on-corpus) | 4 | 1 (`litellm-gateway`, 0.481) | 2 | Answer extracted from the 1 graded doc |
| "How do I bake sourdough?" (off-corpus) | 4 | 0 | 2 | Abstains: no relevant context |

Note on the default run: only one document clears the grade threshold, which
is below `MIN_GRADED = 2`, so the loop spends both rewrites and then answers
best-effort from the single graded doc. The termination bound, not the grade
count, is what ends the loop. That is the behavior the tests in
`tests/test_graph.py` pin down.

## Run 1: default question

```
$ python run.py
Indexing corpus into Qdrant local mode (.../qdrant_data)...
Embedder: sentence-transformers:all-MiniLM-L6-v2 (fallback=False)
Indexed 7 docs.

======================================================================
TRACE
======================================================================
[retrieve] query='How do I route model calls through one gateway so I can swap providers easily?' -> 4 hits: litellm-gateway (0.481), mcp-basics (0.2), slm-local (0.157), langgraph-state (0.078)
[grade] litellm-gateway: similarity=0.481 -> keep (threshold 0.3)
[grade] mcp-basics: similarity=0.2 -> drop (threshold 0.3)
[grade] slm-local: similarity=0.157 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.078 -> drop (threshold 0.3)
[rewrite] attempt 1: 'How do I route model calls through one gateway so I can swap providers easily?' -> 'How do I route model calls through one gateway so I can swap providers easily?'
[retrieve] query='How do I route model calls through one gateway so I can swap providers easily?' -> 4 hits: litellm-gateway (0.481), mcp-basics (0.2), slm-local (0.157), langgraph-state (0.078)
[grade] litellm-gateway: similarity=0.481 -> keep (threshold 0.3)
[grade] mcp-basics: similarity=0.2 -> drop (threshold 0.3)
[grade] slm-local: similarity=0.157 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.078 -> drop (threshold 0.3)
[rewrite] attempt 2: 'How do I route model calls through one gateway so I can swap providers easily?' -> 'How do I route model calls through one gateway so I can swap providers easily?'
[retrieve] query='How do I route model calls through one gateway so I can swap providers easily?' -> 4 hits: litellm-gateway (0.481), mcp-basics (0.2), slm-local (0.157), langgraph-state (0.078)
[grade] litellm-gateway: similarity=0.481 -> keep (threshold 0.3)
[grade] mcp-basics: similarity=0.2 -> drop (threshold 0.3)
[grade] slm-local: similarity=0.157 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.078 -> drop (threshold 0.3)
[generate] extractive fallback from 1 graded docs

======================================================================
FINAL ANSWER
======================================================================
Question: How do I route model calls through one gateway so I can swap providers easily?

Answer (extracted from graded docs):
- LiteLLM acts as a single gateway in front of many model providers. [Routing model calls through a LiteLLM gateway]
- Swapping providers becomes a config change, not a code change, because the request shape stays constant. [Routing model calls through a LiteLLM gateway]
- It also gives you one place for logging, per-key budgets, and fallbacks when a provider is down. [Routing model calls through a LiteLLM gateway]
```

## Run 2: off-corpus question (the failure path)

```
$ python run.py "How do I bake sourdough?"
Indexing corpus into Qdrant local mode (.../qdrant_data)...
Embedder: sentence-transformers:all-MiniLM-L6-v2 (fallback=False)
Indexed 7 docs.

======================================================================
TRACE
======================================================================
[retrieve] query='How do I bake sourdough?' -> 4 hits: scheduled-agents (0.026), grade-retrieval (0.008), langgraph-state (0.005), qdrant-local (-0.006)
[grade] scheduled-agents: similarity=0.026 -> drop (threshold 0.3)
[grade] grade-retrieval: similarity=0.008 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.005 -> drop (threshold 0.3)
[grade] qdrant-local: similarity=-0.006 -> drop (threshold 0.3)
[rewrite] attempt 1: 'How do I bake sourdough?' -> 'How do I bake sourdough?'
[retrieve] query='How do I bake sourdough?' -> 4 hits: scheduled-agents (0.026), grade-retrieval (0.008), langgraph-state (0.005), qdrant-local (-0.006)
[grade] scheduled-agents: similarity=0.026 -> drop (threshold 0.3)
[grade] grade-retrieval: similarity=0.008 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.005 -> drop (threshold 0.3)
[grade] qdrant-local: similarity=-0.006 -> drop (threshold 0.3)
[rewrite] attempt 2: 'How do I bake sourdough?' -> 'How do I bake sourdough?'
[retrieve] query='How do I bake sourdough?' -> 4 hits: scheduled-agents (0.026), grade-retrieval (0.008), langgraph-state (0.005), qdrant-local (-0.006)
[grade] scheduled-agents: similarity=0.026 -> drop (threshold 0.3)
[grade] grade-retrieval: similarity=0.008 -> drop (threshold 0.3)
[grade] langgraph-state: similarity=0.005 -> drop (threshold 0.3)
[grade] qdrant-local: similarity=-0.006 -> drop (threshold 0.3)
[generate] extractive fallback from 0 graded docs

======================================================================
FINAL ANSWER
======================================================================
Question: How do I bake sourdough?

I could not find relevant context in the corpus for this question.
```

Bounded retries, no silent failure, no invented answer.
