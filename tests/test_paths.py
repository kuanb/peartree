import json
import os

import geopandas as gpd
import networkx as nx
import partridge as ptg
import pytest
import numpy as np
from peartree.graph import InsufficientSummaryResults
from peartree.paths import (InvalidGTFS, InvalidTimeBracket,
                            get_representative_feed, load_feed_as_graph,
                            load_synthetic_network_as_graph)
from peartree.toolkit import generate_random_name
from peartree.utilities import config

# Make sure the we set logger on to test logging utilites
# as well, related to each test
config(log_console=True)


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


def test_parsing_when_just_on_trip_during_target_window():
    path = fixture('highdesertpointorus-2018-03-20.zip')
    feed = get_representative_feed(path)

    start = 7*60*60  # 7:00 AM
    end = 8*60*60  # 10:00 AM
    G = load_feed_as_graph(feed, start, end)
    assert len(list(G.nodes())) == 2
    assert len(list(G.edges())) == 1


def test_synthetic_network():
    # Load in the GeoJSON as a JSON and convert to a dictionary
    geojson_path = fixture('synthetic_east_bay.geojson')
    with open(geojson_path, 'r') as gjf:
        reference_geojson = json.load(gjf)

    G1 = load_synthetic_network_as_graph(reference_geojson)

    # This fixture gets broken into 15 chunks, so 15 + 1 = 16
    nodes = list(G1.nodes())
    assert len(nodes) == 16

    # And since it is one-directional, it gets the same edges as chunks
    edges = list(G1.edges())
    assert len(edges) == 15

    # Since this is a one-way graph, with no other context, the
    # graph will be weakly connected
    assert nx.is_strongly_connected(G1) is False

    # Go back to the GeoJSON and set optional bidirectional flag
    for i in range(len(reference_geojson['features'])):
        reference_geojson['features'][i]['properties']['bidirectional'] = True

    G2 = load_synthetic_network_as_graph(reference_geojson)

    # We re-use the same stop nodes for both directions
    nodes = list(G2.nodes())
    assert len(nodes) == 16

    # Double the number of edges as before
    edges = list(G2.edges())
    assert len(edges) == 15 * 2

    # But now, by asking for a bidirectional graph, we can assert strong
    assert nx.is_strongly_connected(G2)


def test_synthetic_network_with_custom_stops():
    # Load in the GeoJSON as a JSON and convert to a dictionary
    geojson_path = fixture('synthetic_east_bay.geojson')
    with open(geojson_path, 'r') as gjf:
        reference_geojson = json.load(gjf)

    # Add in specific, custom stops under new properties key
    custom_stops = [[-122.29225158691406,37.80876678753658],
                    [-122.28886127471924,37.82341261847038],
                    [-122.2701072692871,37.83005652796547]]
    reference_geojson['features'][0]['properties']['stops'] = custom_stops

    G1 = load_synthetic_network_as_graph(reference_geojson)

    # Sanity check the outputs against the custom stops input
    assert len(list(G1.nodes())) == (len(custom_stops) + 2)
    assert len(list(G1.edges())) == (len(custom_stops) + 1)

    # Go back to the GeoJSON and set optional bidirectional flag
    reference_geojson['features'][0]['properties']['bidirectional'] = True

    G2 = load_synthetic_network_as_graph(reference_geojson)

    # We re-use the same stop nodes for both directions
    nodes = list(G2.nodes())
    assert len(nodes) == (len(custom_stops) + 2)

    # Double the number of edges as before
    edges = list(G2.edges())
    assert len(edges) == (len(custom_stops) + 1) * 2

    # But now, by asking for a bidirectional graph, we can assert strong
    assert nx.is_strongly_connected(G2)


def test_feed_edge_types():
    path = fixture('samtrans-2017-11-28.zip')
    feed = get_representative_feed(path)

    start = 7 * 60 * 60
    end = 10 * 60 * 60
    G1 = load_feed_as_graph(feed, start, end)

    # In the base case, all should be transit
    for _, _, e in G1.edges(data=True):
        assert e['mode'] == 'transit'

    # Now perform a second check where we impute walk edges
    G2 = load_feed_as_graph(
        feed, start, end, impute_walk_transfers=True)

    # Count the number of edge types by mode, which should now
    # include walk edges as well
    transit_count = 0
    walk_count = 0
    for _, _, e in G2.edges(data=True):
        if e['mode'] == 'transit':
            transit_count += 1
        if e['mode'] == 'walk':
            walk_count += 1

    # And make sure the correct number were made
    assert transit_count == 1940
    assert walk_count == 864


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


def test_feeds_with_no_direction_id():
    path = fixture('samtrans-2017-11-28.zip')
    feed = get_representative_feed(path)

    # Overwrite the direction id columns in trips df to be nan
    feed.trips['direction_id'] = np.nan

    start = 7 * 60 * 60
    end = 10 * 60 * 60
    G = load_feed_as_graph(feed, start, end)

    # Make sure each node has numeric boarding cost
    for i, node in G.nodes(data=True):
        assert not np.isnan(node['boarding_cost'])