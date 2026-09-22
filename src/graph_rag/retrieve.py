"""Graph RAG retrieval: link a question to entities, then traverse the graph.

The point of a graph over flat chunks: multi-hop questions. If A relates to B
and B to C, no single text chunk mentions A and C together, so vector search
misses the link. Walking the graph finds the A -> B -> C path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx

from .extract import Extractor, Triple, extract_triples
from .graph import build_graph, central_entities, k_hop, path_between, stats


@dataclass
class GraphResult:
    found: bool
    seeds: List[str] = field(default_factory=list)
    path: Optional[List[Tuple[str, str, str]]] = None
    context: List[Tuple[str, str, str]] = field(default_factory=list)
    message: str = ""

    def render(self) -> str:
        if self.path:
            parts = [self.path[0][0]]
            for _u, rel, v in self.path:
                parts.append(f" -[{rel}]-> {v}")
            return "".join(parts)
        return "; ".join(f"{s} {r} {o}" for s, r, o in self.context) or self.message


class GraphRAG:
    def __init__(self, extractor: Extractor = None) -> None:
        self.extractor = extractor
        self.graph = nx.MultiDiGraph()

    def add_text(self, text: str) -> int:
        triples = extract_triples(text, self.extractor)
        for s, r, o in triples:
            self.graph.add_edge(s, o, relation=r)
        return len(triples)

    def ingest_directory(self, path: str, patterns=(".txt", ".md")) -> int:
        total = 0
        for file in sorted(Path(path).rglob("*")):
            if file.suffix.lower() in patterns and file.is_file():
                total += self.add_text(file.read_text(encoding="utf-8"))
        return total

    def link_entities(self, query: str) -> List[str]:
        q = query.lower()
        matches = []
        for n in self.graph.nodes:
            m = re.search(rf"\b{re.escape(str(n).lower())}\b", q)
            if m:
                matches.append((m.start(), -len(str(n)), str(n)))
        matches.sort()  # by position in the query, longer names first on ties
        seen, out = set(), []
        for _pos, _neg_len, n in matches:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def connect(self, a: str, b: str):
        return path_between(self.graph, a, b)

    def search(self, query: str, hops: int = 2) -> GraphResult:
        seeds = self.link_entities(query)
        if not seeds:
            return GraphResult(found=False, message="No known entity found in the question.")
        if len(seeds) >= 2:
            path = self.connect(seeds[0], seeds[1])
            if path:
                return GraphResult(found=True, seeds=seeds, path=path,
                                   context=[(u, r, v) for u, r, v in path])
        context = k_hop(self.graph, seeds, k=hops)
        return GraphResult(found=bool(context), seeds=seeds, context=context,
                           message="" if context else "No connected facts found.")

    def stats(self) -> dict:
        return stats(self.graph)

    def central(self, top: int = 5):
        return central_entities(self.graph, top)
