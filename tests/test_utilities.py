import os

from peartree.paths import get_representative_feed, load_feed_as_graph
from peartree.utilities import graph_from_zip, log, save_graph_to_zip


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_log():
    log('foo')


def test_save_and_read_zip():
    path_1 = fixture('caltrain-2017-07-24.zip')
    feed_1 = get_representative_feed(path_1)

    start = 7 * 60 * 60
    end = 10 * 60 * 60

    G1 = load_feed_as_graph(feed_1, start, end, 'foo')

    # Get counts as a measure to compare with save-read results
    nodes_len_g1 = len(list(G1.nodes()))
    edges_len_g1 = len(list(G1.edges()))

    # First save the graph to a zip
    zip_fpath = 'foobar.zip'
    save_graph_to_zip(G1, zip_fpath)

    # Then read in as a new graph
    G2 = graph_from_zip(zip_fpath)

    # Also immediately remove the zip file so it's not hanging
    # around or impacting later tests
    os.remove(zip_fpath)

    # Get new lengths
    nodes_len_g2 = len(list(G2.nodes()))
    edges_len_g2 = len(list(G2.edges()))

    # They should both be the same as the ones from G1
    assert nodes_len_g1 == nodes_len_g2
    assert edges_len_g1 == edges_len_g2

    # Make sure same numbers of unique nodes are present
    set_n1 = set(list(G1.nodes()))
    set_n2 = set(list(G2.nodes()))
    assert len(set_n1) == len(set_n2)

    # Make sure that all nodes are accounted for
    for n in set_n1:
        assert n in set_n2

    # Do the same for the edges
    e1 = list(G1.edges())
    e2 = list(G2.edges())
    for edge_pair in e1:
        assert edge_pair in e2

    # Also make sure the basic attributes are preserved
    for node_id, node in G2.nodes(data=True):
        for key in ['boarding_cost', 'x', 'y']:
            assert key in node.keys()

    for from_id, to_id, edge in G2.edges(data=True):
        for key in ['length', 'mode']:
            assert key in edge.keys()
