import json
import os

from peartree.graph_tool import _import_graph_tool, nx_to_gt
from peartree.paths import load_synthetic_network_as_graph
from peartree.utilities import config

# Make sure the we set logger on to test logging utilites
# as well, related to each test
config(log_console=True)


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_conversion_to_graph_tool():
    # To be quick, load in the synthetic graph geojson
    geojson_path = fixture('synthetic_san_bruno.geojson')
    with open(geojson_path, 'r') as gjf:
        reference_geojson = json.load(gjf)

    # Then load it onto the graph, as well
    G = load_synthetic_network_as_graph(reference_geojson)

    # For now, just run the operation and ensure it completes as intended
    gtG = nx_to_gt(G)

    # Make sure that the result is indeed a graph-tool Graph class object
    gt = _import_graph_tool()
    assert isinstance(gtG, gt.Graph)

    # Also make sure that the attributes for all parameters have been preserved
    assert set(gtG.vp.keys()) == set(('boarding_cost', 'id', 'x', 'y'))
    assert set(gtG.gp.keys()) == set(('crs', 'name'))
    assert set(gtG.ep.keys()) == set(('length', 'mode'))

    # Make sure the edge count is the same as the NetworkX graph
    nx_edges_len = len([x for x in G.edges()])
    gt_edges_len = len([x for x in gtG.ep['length']])
    assert nx_edges_len == gt_edges_len

    # And same for the vertices count
    nx_nodes_len = len([x for x in G.nodes()])
    gt_nodes_len = len([x for x in gtG.vp['id']])
    assert nx_nodes_len == gt_nodes_len
