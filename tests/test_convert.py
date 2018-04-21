import os

from peartree.paths import get_representative_feed, load_feed_as_graph
from peartree.convert import convert_to_digraph


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_convert_multidigraph_to_digraph():
    path = fixture('samtrans-2017-11-28.zip')
    feed = get_representative_feed(path)

    # Shorter amount of time to speed up the test
    start = 7 * 60 * 60
    end = 8 * 60 * 60
    Gmdg = load_feed_as_graph(feed, start, end, name='foobar')

    # Run conversaion operation
    Gdg = convert_to_digraph(Gmdg)

    assert isinstance(Gdg, nx.DiGraph)
    assert len(Gdg.edges()) == len(Gmdg.edges())
    assert len(Gdg.nodes()) == len(Gmdg.nodes())
