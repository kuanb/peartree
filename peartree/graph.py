from typing import Dict, Union

import networkx as nx
import pandas as pd
import partridge as ptg
from fiona import crs

from .settings import WGS84
from .summarizer import (generate_edge_and_wait_values,
                         generate_summary_edge_costs,
                         generate_summary_wait_times)
from .toolkit import generate_graph_node_dataframe, get_nearest_node


def generate_empty_md_graph(name: str,
                            init_crs: Dict=crs.from_epsg(WGS84)):
    return nx.MultiDiGraph(name=name, crs=init_crs)


def nameify_stop_id(name, sid):
    name = str(name)
    sid = str(sid)
    return '{}_{}'.format(name, sid)


def generate_summary_graph_elements(feed: ptg.gtfs.feed,
                                    target_time_start: int,
                                    target_time_end: int):
    (all_edge_costs,
     all_wait_times) = generate_edge_and_wait_values(feed,
                                                     target_time_start,
                                                     target_time_end)

    summary_edge_costs = generate_summary_edge_costs(all_edge_costs)
    wait_times_by_stop = generate_summary_wait_times(all_wait_times)

    return (summary_edge_costs, wait_times_by_stop)


def generate_cross_feed_edges(G,
                              feed,
                              wait_times_by_stop,
                              connection_threshold):
    # Terminate this process early if the graph is empty
    if (G.number_of_nodes() == 0):
        return pd.DataFrame({'stop_id': [],
                             'to_nodes': [],
                             'edge_costs': []})

    # First, we need a DataFrame representation of the nodes in the graph
    node_df = generate_graph_node_dataframe(G)

    stop_ids = []
    to_nodes = []
    edge_costs = []

    # TODO: Repeating this in populate_graph as well, there may
    #       be a way to condense these two steps well
    for i, row in wait_times_by_stop.iterrows():
        sid = str(row.stop_id)

        # TODO: Join tables before hand to make
        #       this part go faster
        id_mask = (feed.stops.stop_id == sid)
        stop_data_head = feed.stops[id_mask].head(1)

        # Handle the possibility that there are no values for that stop
        # id in the feed subset of wait times
        if not len(stop_data_head):
            continue

        # Once check has cleared, pull out the first row as a pd.Series
        stop_data = stop_data_head.T.squeeze()

        # Ensure that each value is typed correctly prior to being
        # fed into the nearest node method
        lat = float(stop_data.stop_lat)
        lon = float(stop_data.stop_lon)
        point = (lat, lon)
        (nn, nn_dist) = get_nearest_node(node_df, point)

        # Only generate a connector edge if it satisfies the
        # meter distance threshold
        if nn_dist < connection_threshold:
            stop_ids.append(sid)
            to_nodes.append(nn)
            edge_costs.append(nn_dist)

    return pd.DataFrame({'stop_id': stop_ids,
                         'to_node': to_nodes,
                         'distance': edge_costs})


def populate_graph(G: nx.MultiDiGraph,
                   name: str,
                   feed: ptg.gtfs.feed,
                   wait_times_by_stop: pd.DataFrame,
                   summary_edge_costs: pd.DataFrame,
                   connection_threshold: Union[int, float]):
    # As we convert stop ids to actual nodes, let's keep track of those names
    # here so that we can reference them when we add connector edges across
    # the various feeds loaded into the graph
    sid_lookup = {}

    for i, row in wait_times_by_stop.iterrows():
        sid = str(row.stop_id)
        full_sid = nameify_stop_id(name, sid)

        # TODO: Join tables before hand to make
        #       this part go faster
        id_mask = (feed.stops.stop_id == sid)
        stop_data_head = feed.stops[id_mask].head(1)

        # Handle the possibility that there are no values for that stop
        # id in the feed subset of wait times
        if not len(stop_data_head):
            continue

        # Once check has cleared, pull out the first row as a pd.Series
        stop_data = stop_data_head.T.squeeze()

        # Add to the lookup crosswalk dictionary
        sid_lookup[sid] = full_sid

        G.add_node(full_sid,
                   boarding_cost=row.avg_cost,
                   y=stop_data.stop_lat,
                   x=stop_data.stop_lon)

    for i, row in summary_edge_costs.iterrows():
        sid_fr = nameify_stop_id(name, row.from_stop_id)
        sid_to = nameify_stop_id(name, row.to_stop_id)
        G.add_edge(sid_fr,
                   sid_to,
                   length=row.edge_cost)

    # Generate cross feed edge values
    cross_feed_edges = generate_cross_feed_edges(G,
                                                 feed,
                                                 wait_times_by_stop,
                                                 connection_threshold)

    # Now add the cross feed edge connectors to the graph to
    # capture transfer points
    for i, row in cross_feed_edges.iterrows():
        # Extract the row column values as discrete variables
        sid = row.stop_id
        to = row.to_node
        d = row.distance

        # Use the lookup table to get converted stop id name
        full_sid = sid_lookup[sid]

        # Convert to km/hour
        kmph = (d / 1000) / 4.5

        # Convert to seconds
        in_seconds = kmph * 60 * 60

        G.add_edge(full_sid, to, length=in_seconds)

    return G
