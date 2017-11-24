import os

import networkx as nx
import partridge as ptg
import pytest
from peartree.paths import (InvalidGTFS, _generate_random_name,
                            get_representative_feed, load_feed_as_graph)


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_generate_name():
    name = _generate_random_name(10)
    assert len(name) == 10

    name = _generate_random_name(12)
    assert len(name) == 12

    name = _generate_random_name()
    assert isinstance(name, str)


def test_empty_feed():
    path = fixture('empty.zip')
    with pytest.raises(InvalidGTFS):
        get_representative_feed(path)


def test_extract_valid_feed():
    # Read in without name, or any
    # other optional arguments
    path = fixture('caltrain-2017-07-24.zip')
    feed = get_representative_feed(path)
    assert isinstance(feed, ptg.gtfs.feed)


def test_feed_to_graph_path():
    path = fixture('caltrain-2017-07-24.zip')
    feed = get_representative_feed(path)

    start = 7 * 60 * 60
    end = 10 * 60 * 60

    G = load_feed_as_graph(feed,
                           start,
                           end,
                           'foo')

    assert isinstance(G, nx.MultiDiGraph)
