#!/usr/bin/env python3
"""Run the LangGraph retrieve -> grade -> generate loop end to end.

Usage:
    ./run.py ["your question here"]

Rebuilds the Qdrant local index from corpus.py, runs the graph, and prints
the node trace plus the final answer. The sample question is chosen to match
the corpus, so grading passes on the first pass.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_graph, build_index

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")

SAMPLE_QUESTION = "How do I route model calls through one gateway so I can swap providers easily?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUESTION
    print(f"Indexing corpus into Qdrant local mode ({DATA_DIR})...")
    client, embedder, used_fallback = build_index(DATA_DIR)
    print(f"Embedder: {embedder.name} (fallback={used_fallback})")
    print(f"Indexed {len(__import__('corpus').DOCS)} docs.\n")

    graph = build_graph(client, embedder)
    result = graph.invoke({
        "question": question,
        "active_query": question,
        "retrieved": [],
        "graded": [],
        "retries": 0,
        "answer": "",
        "trace": [],
    })
    print("=" * 70)
    print("TRACE")
    print("=" * 70)
    for line in result["trace"]:
        print(line)
    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["answer"])


if __name__ == "__main__":
    main()
