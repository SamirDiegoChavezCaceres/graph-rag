"""Graph RAG walkthrough: build a graph, answer a multi-hop question, and show
why flat vector RAG misses it.

    python scripts/demo.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from graph_rag import GraphRAG

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

_STOP = frozenset("how is are the a an of to in on at and or do does related".split())


def sentences():
    """Every sentence in the corpus, the unit a chunk retriever would index."""
    out = []
    for file in sorted(CORPUS.glob("*")):
        for s in re.split(r"(?<=[.!?])\s+", file.read_text(encoding="utf-8").strip()):
            if s.strip():
                out.append(s.strip())
    return out


def flat_retrieve(query, chunks, k=3):
    """Offline, dependency-free lexical retrieval (token overlap), so the 'flat
    RAG' baseline runs anywhere. It ranks chunks by shared words with the query."""
    q = {w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if w not in _STOP}
    scored = [(c, len(q & set(re.findall(r"[A-Za-z0-9]+", c.lower())))) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:k]]


def make_answerer():
    """Return a grounded answer(query, context) via OpenAI, or None when offline."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    client = OpenAI()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def answer(query: str, context: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[
                {"role": "system", "content":
                 "Answer using ONLY the facts. Connect the two things in the question "
                 "by chaining the facts step by step, even if the link is indirect. If "
                 "the facts do not connect them at all, say you cannot tell from the "
                 "given facts. Answer in one sentence on a single line."},
                {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    return answer


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

    rule("3. No graph (flat retrieval) vs graph, same question")
    top = flat_retrieve(q, sentences(), k=3)
    print("  flat RAG retrieves the 3 passages most similar to the question:")
    for c in top:
        print(f"    - {c}")
    print("  the bridge 'Acme acquired Beta' is not retrieved (it names neither")
    print("  Ada nor Orion), so flat context cannot connect them.")
    answer = make_answerer()
    if answer:
        flat_ctx = "\n".join(top)
        graph_ctx = "; ".join(f"{s} {r.replace('_', ' ')} {o}" for s, r, o in res.path)
        print(f"  no graph -> {answer(q, flat_ctx)}")
        print(f"  graph    -> {answer(q, graph_ctx)}")
    else:
        print(f"  graph path -> {res.render()}")
        print("  (set OPENAI_API_KEY to also generate both answers)")

    rule("4. Neighborhood query")
    q2 = "What do we know around Beta?"
    print(f"  Q: {q2}")
    res2 = rag.search(q2, hops=1)
    for s, r, o in res2.context:
        print(f"    {s} -[{r}]-> {o}")


if __name__ == "__main__":
    main()
