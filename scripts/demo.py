"""Graph RAG walkthrough: build a graph, answer a multi-hop question, and show
why flat vector RAG misses it.

    python scripts/demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from graph_rag import GraphRAG

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def pick_extractor():
    """The real path is LLM extraction; use it when a key is set, else the
    offline rule-based extractor so the demo still runs."""
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from graph_rag import LLMExtractor

            return LLMExtractor(), "LLMExtractor (OpenAI)"
        except Exception:
            pass
    return None, "RuleBasedExtractor (offline)"


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    extractor, name = pick_extractor()
    rag = GraphRAG(extractor=extractor) if extractor else GraphRAG()
    print(f"extractor: {name}")
    n = rag.ingest_directory(str(CORPUS))

    rule("1. Build the knowledge graph")
    print(f"  triples extracted: {n} | {rag.stats()}")
    print("  most central entities:",
          ", ".join(f"{e} ({s:.2f})" for e, s in rag.central(3)))

    rule("2. Multi-hop question (the point of a graph)")
    q = "How is Ada related to Orion?"
    print(f"  Q: {q}")
    res = rag.search(q)
    print(f"  seeds: {res.seeds}")
    print(f"  path : {res.render()}")

    rule("3. Why flat vector RAG struggles here (shown, not claimed)")
    for file in sorted(CORPUS.glob("*")):
        text = file.read_text(encoding="utf-8")
        print(f"  {file.name:14} Ada={'Ada' in text}  Orion={'Orion' in text}")
    print("  -> no single document mentions both, so a chunk retriever has no")
    print(f"     passage that links them. The graph does: {res.render()}")

    rule("4. Neighborhood query")
    q2 = "What do we know around Beta?"
    print(f"  Q: {q2}")
    res2 = rag.search(q2, hops=1)
    for s, r, o in res2.context:
        print(f"    {s} -[{r}]-> {o}")


if __name__ == "__main__":
    main()
