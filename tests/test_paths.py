import json
import os

import geopandas as gpd
import networkx as nx
import partridge as ptg
import pytest
from peartree.graph import InsufficientSummaryResults
from peartree.paths import (InvalidGTFS, InvalidTimeBracket,
                            get_representative_feed, load_feed_as_graph,
                            load_synthetic_network_as_graph)
from peartree.toolkit import generate_random_name


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
    name = generate_random_name(10)
    assert len(name) == 10

    name = generate_random_name(12)
    assert len(name) == 12

    name = generate_random_name()
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


def test_synthetic_network():
    # Load in the GeoJSON as a JSON and convert to a dictionary
    geojson_path = fixture('synthetic_east_bay.geojson')
    with open(geojson_path, 'r') as gjf:
        reference_geojson = json.load(gjf)

    G = load_synthetic_network_as_graph(reference_geojson)

    # This fixture gets broken into 15 chunks, so 15 + 1 = 16
    nodes = list(G.nodes())
    assert len(nodes) == 16

    # And since it is one-directional, it gets the same edges as chunks
    edges = list(G.edges())
    assert len(edges) == 15


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
    orig_node_list = list(G.nodes())

    path_2 = fixture('samtrans-2017-11-28.zip')
    feed_2 = get_representative_feed(path_2)
    G = load_feed_as_graph(feed_2, start, end, 'bar', G)

    assert isinstance(G, nx.MultiDiGraph)
    _check_unreasonable_lengths(G, max_reasonable_segment_length)

    # Part 2 of sanity check that the number of nodes and edges go up
    node_len_2 = len(G.nodes())
    edge_len_2 = len(G.edges())
    assert node_len_2 > orig_node_len
    assert edge_len_2 > orig_edge_len

    connector_edge_count = 0
    for from_node, to_node, edge in G.edges(data=True):
        # Make sure that a length measure has been calculated for each
        # edge in the resulting graph, also sanity check that all are
        # positive values
        assert 'length' in edge.keys()
        assert isinstance(edge['length'], float)
        assert edge['length'] >= 0

        # Also, we should also make sure that edges were also created that
        # connect the two feeds
        from_orig_a = from_node in orig_node_list
        from_orig_b = to_node in orig_node_list
        one_valid_fr = from_orig_a and (not from_orig_b)
        one_valid_to = (not from_orig_a) and from_orig_b
        if one_valid_fr or one_valid_to:
            connector_edge_count += 1

    # We know that there should be 9 new edges that are created to connect
    # the two GTFS feeds in the joint graph
    assert connector_edge_count == 9

    # Now reload in the synthetic graph geojson
    geojson_path = fixture('synthetic_san_bruno.geojson')
    with open(geojson_path, 'r') as gjf:
        reference_geojson = json.load(gjf)

    # Then load it onto the graph, as well
    G = load_synthetic_network_as_graph(reference_geojson, existing_graph=G)

    # And make sure it connected correctly
    node_len_3 = len(G.nodes())
    edge_len_3 = len(G.edges())
    assert node_len_3 - node_len_2 == 74
    assert edge_len_3 - edge_len_2 == 80
