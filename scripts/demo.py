"""Graph RAG walkthrough: build a graph, answer a multi-hop question, and show
why flat vector RAG misses it.

    python scripts/demo.py
"""

from __future__ import annotations

from pathlib import Path

from graph_rag import GraphRAG

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    rag = GraphRAG()
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

    rule("3. Why flat vector RAG struggles here")
    print("  No single sentence mentions both 'Ada' and 'Orion', so chunk")
    print("  similarity never links them. The graph walks Ada -> ... -> Orion.")

    rule("4. Neighborhood query")
    q2 = "What do we know around Beta?"
    print(f"  Q: {q2}")
    res2 = rag.search(q2, hops=1)
    for s, r, o in res2.context:
        print(f"    {s} -[{r}]-> {o}")


if __name__ == "__main__":
    main()
