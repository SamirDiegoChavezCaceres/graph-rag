"""Build and query a knowledge graph from triples (networkx)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from .extract import Triple


def build_graph(triples: List[Triple]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for s, r, o in triples:
        g.add_edge(s, o, relation=r)
    return g


def stats(g: nx.MultiDiGraph) -> dict:
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}


def central_entities(g: nx.MultiDiGraph, top: int = 5) -> List[Tuple[str, float]]:
    pr = nx.pagerank(nx.DiGraph(g)) if g.number_of_edges() else {}
    return sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:top]


def _relation_between(g: nx.MultiDiGraph, u: str, v: str) -> Tuple[str, str]:
    """Return (relation, direction) for the edge connecting u and v either way."""
    data = g.get_edge_data(u, v)
    if data:
        return list(data.values())[0]["relation"], "->"
    data = g.get_edge_data(v, u)
    if data:
        return list(data.values())[0]["relation"], "<-"
    return "related_to", "--"


def path_between(g: nx.MultiDiGraph, src: str, dst: str) -> Optional[List[Tuple[str, str, str]]]:
    """Shortest connecting path as a list of (from, relation, to) steps.

    Uses the undirected view so a chain connects regardless of edge direction.
    """
    ug = g.to_undirected(as_view=True)
    if src not in ug or dst not in ug or not nx.has_path(ug, src, dst):
        return None
    nodes = nx.shortest_path(ug, src, dst)
    steps = []
    for u, v in zip(nodes, nodes[1:]):
        rel, direction = _relation_between(g, u, v)
        steps.append((u, rel if direction != "<-" else f"{rel} (of)", v))
    return steps


def k_hop(g: nx.MultiDiGraph, seeds: List[str], k: int = 1) -> List[Tuple[str, str, str]]:
    """All triples within k hops of any seed (undirected expansion)."""
    ug = g.to_undirected(as_view=True)
    reach = set()
    for s in seeds:
        if s in ug:
            reach |= set(nx.ego_graph(ug, s, radius=k).nodes)
    return [(u, d["relation"], v) for u, v, d in g.edges(data=True)
            if u in reach and v in reach]
