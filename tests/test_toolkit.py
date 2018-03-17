import networkx as nx
import pytest
from peartree.toolkit import reproject


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

    expected_xs = [-1932968.345, -1932884.818]
    for x, ex in zip(xs, expected_xs):
        assert x == pytest.approx(ex, abs=0.01)

    expected_ys = [-543020.339, -543361.855]
    for y, ey in zip(ys, expected_ys):
        assert y == pytest.approx(ey, abs=0.01)
