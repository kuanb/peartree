from peartree.graph import generate_empty_md_graph


def test_generate_empty_graph():
    G = generate_empty_md_graph('foo')
    assert len(G.edges()) == 0
    assert len(G.nodes()) == 0
