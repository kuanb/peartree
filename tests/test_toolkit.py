import os

import networkx as nx
import pytest
from peartree.paths import get_representative_feed, load_feed_as_graph
from peartree.toolkit import coalesce, reproject, simplify_graph


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def _dict_equal(got, want):
    all_valid = True
    for key in want:
        if not got[key] == want[key]:
            all_valid = False

    return all_valid


def test_feed_to_graph_plot():
    # Create a simple graph
    G = nx.Graph(crs={'init': 'epsg:4326', 'no_defs': True})

    # And add two nodes to it
    G.add_node('a', x=-122.2729918, y=37.7688136)
    G.add_node('b', x=-122.2711039, y=37.7660709)

    # Now, reproject the graph to the default 2163 epsg
    G2 = reproject(G)

    # Now make sure the projection has been retained during simplification
    assert 'crs' in G.graph
    assert G.graph['crs']['init'] == 'epsg:4326'
    assert 'crs' in G2.graph
    assert G2.graph['crs']['init'] == 'epsg:2163'

    # Extract the x and y values of the reproject nodes
    xs = []
    ys = []
    for i, node in G2.nodes(data=True):
        xs.append(node['x'])
        ys.append(node['y'])

    # TODO: Can we improve how this is assessed? Seems like G.nodes()
    #       does not return nodes in same order in Py 3.6 as in 3.5
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
    G = nx.MultiDiGraph(crs={'init': 'epsg:4326', 'no_defs': True}, name='foo')

    # And add two nodes to it
    G.add_node('a', x=-122.2729918, y=37.7688136, boarding_cost=10)
    G.add_node('b', x=-122.2711039, y=37.7660709, boarding_cost=15)
    G.add_node('c', x=-122.2711038, y=37.7660708, boarding_cost=12)
    G.add_edge('a', 'b', length=10, mode='transit')
    G.add_edge('b', 'c', length=1, mode='walk')

    # Also add a node, and edge that is a more expensive variant
    # of effectively the same edge
    G.add_node('b_alt', x=-122.2711039, y=37.7660709, boarding_cost=13.5)
    G.add_edge('a', 'b_alt', length=100, mode='transit')

    # Also add a second edge between the same nodes, but with smaller weight
    G.add_edge('a', 'b_alt', length=10, mode='transit')

    G2 = reproject(G)
    G2c = coalesce(G2, 200)

    # Same akward situation as before, where edges are returned in
    # different order between Py 3.5 and 3.6
    for i, node in G2c.nodes(data=True):
        a = _dict_equal(node, {
            'x': -1933000,
            'y': -543000,
            'boarding_cost': 10.0})
        b = _dict_equal(node, {
            'x': -1932800,
            'y': -543400,
            'boarding_cost': 13.5})
        assert (a or b)

    all_edges = list(G2c.edges(data=True))
    assert len(all_edges) == 1

    # Make sure that the one edge came out as expected
    assert _dict_equal(all_edges[0][2], {'length': 55, 'mode': 'transit'})


def test_simplify_graph():
    path = fixture('samtrans-2017-11-28.zip')
    feed = get_representative_feed(path)

    # Shorter amount of time to speed up the test
    start = 7 * 60 * 60
    end = 8 * 60 * 60
    G = load_feed_as_graph(feed, start, end, name='foobar')

    # Run simplification
    Gs = simplify_graph(G)

    # TODO: We have this ongoing issue where we can't
    #       consistently test by index for edges, so we need
    #       to figure out _how_ to test for a specific edge
    assert len(Gs.nodes()) == 298
    assert len(Gs.edges()) == 451

    # Pull out a summary list of edges as dicts
    all_es = []
    for e_fr, e_to, edge in Gs.edges(data=True):
        edge['from'] = e_fr
        edge['to'] = e_to

        # Let's just look at those that have a larger
        # length associated with them and were coalesced from
        # other internal ways (so a geometry object is present)
        if edge['length'] > 110 and 'geometry' in edge.keys():
            all_es.append(edge)

    # Sort the list and pull the max out, where max is determined
    # based on the number of coordinates in the LineString
    target_edge = max(all_es, key=lambda x: len(x['geometry'].coords.xy[0]))
    assert target_edge['length'] == 5114.0
    assert target_edge['mode'] == 'transit'
    assert target_edge['from'] == 'foobar_351008'
    assert target_edge['to'] == 'foobar_334008'
    assert len(target_edge['geometry'].coords.xy[0]) == 49
