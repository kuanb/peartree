import os
from time import time

from peartree.graph import generate_empty_md_graph, populate_graph
from peartree.paths import get_representative_feed
from peartree.summarizer import (generate_edge_and_wait_values,
                                 generate_summary_edge_costs,
                                 generate_summary_wait_times)


def fixture(filename):
    return os.path.join('tests', 'fixtures', filename)


def test_feed_to_graph_performance():
    # Replicate the original workflow of the graph creation path
    # but open up to expose to benchmarking/performance profiling
    start = 7 * 60 * 60
    end = 10 * 60 * 60

    print('Running time profiles on each major '
          'function in graph generation workflow')

    a = time()
    path = fixture('samtrans-2017-11-28.zip')
    feed = get_representative_feed(path)
    elapsed = round(time() - a, 2)
    print(f'Perf of get_representative_feed: {elapsed}s')

    fl = len(feed.routes)
    print(f'Iteration on {fl} routes.')

    a = time()
    (all_edge_costs,
     all_wait_times) = generate_edge_and_wait_values(feed,
                                                     start,
                                                     end)
    elapsed = round(time() - a, 2)
    print(f'Perf of generate_edge_and_wait_values: {elapsed}s')

    a = time()
    summary_edge_costs = generate_summary_edge_costs(all_edge_costs)
    elapsed = round(time() - a, 2)
    print(f'Perf of generate_summary_edge_costs: {elapsed}s')

    a = time()
    wait_times_by_stop = generate_summary_wait_times(all_wait_times)
    elapsed = round(time() - a, 2)
    print(f'Perf of generate_summary_wait_times: {elapsed}s')

    a = time()
    G = generate_empty_md_graph('foo')
    elapsed = round(time() - a, 2)
    print(f'Perf of generate_empty_md_graph: {elapsed}s')

    a = time()
    G = populate_graph(G,
                       'bar',
                       feed,
                       wait_times_by_stop,
                       summary_edge_costs,
                       50,
                       4.5)
    elapsed = round(time() - a, 2)
    print(f'Perf of populate_graph: {elapsed}s')
