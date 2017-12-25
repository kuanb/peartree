import os

from peartree.graph import (generate_empty_md_graph,
                            generate_summary_graph_elements)
from peartree.paths import get_representative_feed


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_generate_empty_graph():
    G = generate_empty_md_graph('foo')
    assert len(G.edges()) == 0
    assert len(G.nodes()) == 0


def test_generate_summary_graph_elements():
    path_1 = fixture('caltrain-2017-07-24.zip')
    feed_1 = get_representative_feed(path_1)

    start = 7 * 60 * 60
    end = 10 * 60 * 60

    (summary_edge_costs,
     wait_times_by_stop) = generate_summary_graph_elements(feed_1, start, end)

    # Ensure that the summary edge cost dataframe looks as it should
    ec_cols = ['edge_cost', 'from_stop_id', 'to_stop_id']
    for c in ec_cols:
        assert c in summary_edge_costs.columns

    # Make sure that all edges are unique - there are no duplicated
    # in the returned edge dataframe (each should be its own summary)
    f = summary_edge_costs.from_stop_id
    t = summary_edge_costs.to_stop_id
    z = list(zip(f, t))
    assert len(list(set(z))) == len(z)

    # Ensure that the wait times dataframe looks as it should
    wt_cols = ['avg_cost', 'stop_id']
    for c in wt_cols:
        assert c in wait_times_by_stop.columns

    # Sanity check edge costs
    mask = (wait_times_by_stop.avg_cost < 0)
    assert len(wait_times_by_stop[mask]) == 0

    # Make sure that there are stop ids unique
    u = wait_times_by_stop.stop_id.unique()
    assert len(u) == len(wait_times_by_stop)
