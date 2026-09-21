# tiny RAG over 7 short AI-infrastructure notes
DOCS = [
    {
        "id": "litellm-gateway",
        "title": "Routing model calls through a LiteLLM gateway",
        "text": (
            "LiteLLM acts as a single gateway in front of many model providers. "
            "Your code sends OpenAI-format requests to the LiteLLM proxy, and the proxy "
            "translates them to Anthropic, OpenAI, Bedrock, Vertex, or a local Ollama server. "
            "Swapping providers becomes a config change, not a code change, because the "
            "request shape stays constant. It also gives you one place for logging, "
            "per-key budgets, and fallbacks when a provider is down."
        ),
    },
    {
        "id": "qdrant-local",
        "title": "Qdrant in local mode for small RAG projects",
        "text": (
            "Qdrant can run in local mode with no server and no Docker: the client library "
            "embeds a storage engine and persists vectors to a directory on disk. "
            "Vectors are indexed with HNSW, an approximate nearest-neighbor structure that "
            "keeps search fast as the collection grows. Each point can carry a payload "
            "dictionary, so you can filter by metadata like doc title or source before "
            "searching. Local mode is enough for a prototype; the same client code works "
            "against a hosted Qdrant cluster when you outgrow it."
        ),
    },
    {
        "id": "langgraph-state",
        "title": "Why LangGraph uses explicit state",
        "text": (
            "LangGraph models a workflow as nodes that read and write a shared state object. "
            "Each node receives the current state and returns the fields it wants to update, "
            "so every step's inputs and outputs are visible in one place. Because state is "
            "explicit, you can pause a run, inspect it, and resume later, which is the basis "
            "for human-in-the-loop checkpoints. A plain sequential chain hides intermediate "
            "values inside closures, so debugging a bad retrieval means rerunning everything."
        ),
    },
    {
        "id": "grade-retrieval",
        "title": "Grading retrieved chunks before generating",
        "text": (
            "A retrieval grader scores each chunk against the question and drops the ones "
            "that are not really relevant. The simplest grader is a cosine-similarity "
            "threshold on the same embeddings used for search. When too few chunks pass, "
            "the graph loops back, rewrites the query, and retrieves again instead of "
            "generating from weak context. This retrieve-grade-retry cycle is the core of "
            "self-RAG style agents: generation only happens once the evidence clears a bar."
        ),
    },
    {
        "id": "slm-local",
        "title": "Running small language models locally",
        "text": (
            "Small language models in the 7 to 8 billion parameter range can run on a single "
            "consumer GPU using quantized formats like GGUF. Runtimes such as llama.cpp and "
            "Ollama load the quantized weights and serve an OpenAI-compatible endpoint on "
            "localhost. Quantization to 4 or 5 bits shrinks the model enough to fit in 8 GB "
            "of VRAM with modest quality loss. Fine-tuning with QLoRA trains small adapter "
            "weights on top of the frozen quantized base model."
        ),
    },
    {
        "id": "scheduled-agents",
        "title": "Scheduled agents and the morning brief pattern",
        "text": (
            "A scheduled agent wakes up on a cron schedule, gathers fresh context, and "
            "produces a short brief. The typical shape is: collect signals from APIs and "
            "local files, rank them by relevance, then render a compact summary. Keeping the "
            "brief short matters more than covering everything, because the reader decides "
            "in seconds whether to keep reading. Idempotency matters too: if the job runs "
            "twice, it should not send the brief twice."
        ),
    },
    {
        "id": "mcp-basics",
        "title": "MCP servers expose tools to agents",
        "text": (
            "The Model Context Protocol defines how an agent discovers and calls external "
            "tools. An MCP server advertises a list of tools with JSON schemas, and the "
            "agent decides which tool to call and with what arguments. The server executes "
            "the tool and returns the result as context for the next model call. This keeps "
            "tool definitions decoupled from any single agent framework."
        ),
    },
]
