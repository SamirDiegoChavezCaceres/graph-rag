from pathlib import Path

from graph_rag import GraphRAG, extract_triples

CORPUS = str(Path(__file__).resolve().parents[1] / "corpus")


def test_rule_based_extraction():
    triples = extract_triples("Ada founded Acme. Acme is based in Lima.")
    assert ("Ada", "founded", "Acme") in triples
    assert ("Acme", "based_in", "Lima") in triples


def _rag():
    rag = GraphRAG()
    rag.ingest_directory(CORPUS)
    return rag


def test_graph_has_expected_entities():
    g = _rag().graph
    for node in ("Ada", "Acme", "Beta", "Orion", "UNSA"):
        assert node in g
    assert g.number_of_edges() >= 8


def test_entity_linking_in_query_order():
    assert _rag().link_entities("How is Ada related to Orion?") == ["Ada", "Orion"]


def test_multi_hop_path():
    res = _rag().search("How is Ada related to Orion?")
    assert res.found and res.path
    nodes = [res.path[0][0]] + [step[2] for step in res.path]
    assert nodes[0] == "Ada" and nodes[-1] == "Orion"
    assert "Acme" in nodes and "Beta" in nodes   # it went through the chain


def test_no_path_for_unknown_entity():
    rag = _rag()
    assert rag.connect("Ada", "Nonexistent") is None
    assert rag.search("Tell me about Nonexistent").found is False


def test_neighborhood_query():
    facts = _rag().search("What is around Beta?", hops=1).context
    assert ("Carlos", "works_at", "Beta") in facts
