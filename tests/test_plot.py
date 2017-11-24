import os

from peartree.paths import get_representative_feed, load_feed_as_graph
from peartree.plot import generate_plot


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_feed_to_graph_path():
    path = fixture('caltrain-2017-07-24.zip')
    feed = get_representative_feed(path)

    start = 7 * 60 * 60
    end = 10 * 60 * 60

    G = load_feed_as_graph(feed, start, end)

    fig, ax = generate_plot(G)
