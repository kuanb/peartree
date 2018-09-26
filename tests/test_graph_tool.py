import pytest

from peartree.graph_tool import nx_to_gt
from peartree.paths import load_synthetic_network_as_graph

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
    nx_to_gt(G)


