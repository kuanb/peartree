import os

import geopandas as gpd
import networkx as nx
import partridge as ptg
import pytest
from peartree.graph import InsufficientSummaryResults
from peartree.paths import (InvalidGTFS, InvalidTimeBracket,
                            _generate_random_name, get_representative_feed,
                            load_feed_as_graph)


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def _check_unreasonable_lengths(G, threshold):
    # Take the edges of the graph and evaluate the impedance
    # calculations
    edges_with_geom = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'length' in data:
            edges_with_geom.append({'u': u,
                                    'v': v,
                                    'key': key,
                                    'length': data['length']})
    gdf_edges = gpd.GeoDataFrame(edges_with_geom)
    mask = gdf_edges['length'] > threshold
    assert len(gdf_edges[mask]) == 0


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


def test_loading_in_too_small_timeframes():
    path_1 = fixture('caltrain-2017-07-24.zip')
    feed_1 = get_representative_feed(path_1)

    # Loading in a time frame that will result
    # in no valid results
    start = 0
    end = 1
    with pytest.raises(InsufficientSummaryResults):
        load_feed_as_graph(feed_1, start, end)


def test_loading_in_invalid_timeframes():
    path_1 = fixture('caltrain-2017-07-24.zip')
    feed_1 = get_representative_feed(path_1)

    # Loading in a timeframe where the
    # start comes before the end
    start = 500
    end = 100
    with pytest.raises(InvalidTimeBracket):
        load_feed_as_graph(feed_1, start, end)

    # Loading in a timeframe is of length 0
    start = 0
    end = 0
    with pytest.raises(InvalidTimeBracket):
        load_feed_as_graph(feed_1, start, end)

    start = 1000
    end = 1000
    with pytest.raises(InvalidTimeBracket):
        load_feed_as_graph(feed_1, start, end)


def test_feed_to_graph_path():
    path_1 = fixture('caltrain-2017-07-24.zip')
    feed_1 = get_representative_feed(path_1)

    start = 7 * 60 * 60
    end = 10 * 60 * 60

    G = load_feed_as_graph(feed_1, start, end, 'foo')

    # We should assume all routes do not have segments that exceed some
    # given length (measured in seconds)
    max_reasonable_segment_length = 60 * 60
    _check_unreasonable_lengths(G, max_reasonable_segment_length)

    # Sanity check that the number of nodes and edges go up
    orig_node_len = len(G.nodes())
    orig_edge_len = len(G.edges())

    path_2 = fixture('samtrans-2017-11-28.zip')
    feed_2 = get_representative_feed(path_2)
    G = load_feed_as_graph(feed_2, start, end, 'bar', G)

    assert isinstance(G, nx.MultiDiGraph)
    _check_unreasonable_lengths(G, max_reasonable_segment_length)

    # Part 2 of sanity check that the number of nodes and edges go up
    new_node_len = len(G.nodes())
    new_edge_len = len(G.edges())
    assert new_node_len > orig_node_len
    assert new_edge_len > orig_edge_len
