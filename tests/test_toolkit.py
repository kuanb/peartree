import networkx as nx
import pytest
from peartree.toolkit import coalesce, reproject


def _assert_dict(got, want):
    for key in want:
        if not got[key] == want[key]:
            print(got)
        assert got[key] == want[key]


def test_feed_to_graph_plot():
    # Create a simple graph
    G = nx.Graph(crs={'init': 'epsg:4326', 'no_defs': True})

    # And add two nodes to it
    G.add_node('a', x=-122.2729918, y=37.7688136)
    G.add_node('b', x=-122.2711039, y=37.7660709)

    # Now, reproject the graph to the default 2163 epsg
    G2 = reproject(G)

    # Extract the x and y values of the reproject nodes
    xs = []
    ys = []
    for i, node in G2.nodes(data=True):
        xs.append(node['x'])
        ys.append(node['y'])

    # TODO: Can we improve how this is assessed? Seems like G.nodes()
    #       does not retun the nodes in the same order each time; there
    #       is a component of the node and edge creation that is not
    #       deterministic and this is a hack to circumvent that issue.
    expected_xs = [-1932968.345, -1932884.818]
    for x in xs:
        a = (x == pytest.approx(expected_xs[0], abs=0.01))
        b = (x == pytest.approx(expected_xs[1], abs=0.01))
        assert (a or b)

    expected_ys = [-543020.339, -543361.855]
    for y in ys:
        a = (y == pytest.approx(expected_ys[0], abs=0.01))
        b = (y == pytest.approx(expected_ys[1], abs=0.01))
        assert (a or b)


def test_coalesce_operation():
    # Create a simple graph
    G = nx.Graph(crs={'init': 'epsg:4326', 'no_defs': True}, name='foo')

    # And add two nodes to it
    G.add_node('a', x=-122.2729918, y=37.7688136, boarding_cost=10)
    G.add_node('b', x=-122.2711039, y=37.7660709, boarding_cost=15)
    G.add_node('c', x=-122.2711038, y=37.7660708, boarding_cost=12)
    G.add_edge('a', 'b', length=10, mode='transit')
    G.add_edge('b', 'c', length=1, mode='walk')

    G2 = reproject(G)

    G2c = coalesce(G2, 200)
    G2c.nodes(data=True), G2c.edges(data=True)

    _assert_dict(G2c.nodes['foo_0'], {
                 'x': -1933000, 'y': -543000, 'boarding_cost': 10.0})
    _assert_dict(G2c.nodes['foo_1'], {
                 'x': -1932800, 'y': -543400, 'boarding_cost': 13.5})

    all_edges = list(G2c.edges(data=True))
    assert len(all_edges) == 1

    # Make sure that the one edge came out as expected
    _assert_dict(all_edges[0][2], {'length': 10, 'mode': 'transit'})
