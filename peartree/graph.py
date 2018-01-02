from typing import Dict, Union

import networkx as nx
import pandas as pd
import partridge as ptg
from fiona import crs

from .settings import WGS84
from .summarizer import (generate_edge_and_wait_values,
                         generate_summary_edge_costs,
                         generate_summary_wait_times)
from .toolkit import generate_graph_node_dataframe, get_nearest_nodes


class InsufficientSummaryResults(Exception):
    pass


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

    # Handle if there are no valid edges returned (or wait times)
    if all_edge_costs is None or len(all_edge_costs) == 0:
        raise InsufficientSummaryResults('The target time frame returned no '
                                         'valid edge costs from feed object.')
    if all_wait_times is None or len(all_wait_times) == 0:
        raise InsufficientSummaryResults('The target time frame returned no '
                                         'valid wait times from feed object.')

    summary_edge_costs = generate_summary_edge_costs(all_edge_costs)
    wait_times_by_stop = generate_summary_wait_times(all_wait_times)

    return (summary_edge_costs, wait_times_by_stop)


def generate_cross_feed_edges(G,
                              stops_df,
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
    for i, row in stops_df.iterrows():
        sid = str(row.stop_id)

        # Ensure that each value is typed correctly prior to being
        # fed into the nearest node method
        lat = float(row.stop_lat)
        lon = float(row.stop_lon)
        point = (lat, lon)
        (nns, nn_dists) = get_nearest_nodes(node_df,
                                            point,
                                            connection_threshold)

        # Iterate through series results and add to output
        for node_id, dist_val in zip(nns, nn_dists):
            stop_ids.append(sid)
            to_nodes.append(node_id)
            edge_costs.append(dist_val)

    return pd.DataFrame({'stop_id': stop_ids,
                         'to_node': to_nodes,
                         'distance': edge_costs})


def _merge_stop_waits_and_attributes(wait_times_by_stop: pd.DataFrame,
                                     feed_stops: pd.DataFrame) -> pd.DataFrame:
    wt_sub = wait_times_by_stop[['avg_cost', 'stop_id']]
    fs_sub = feed_stops[['stop_lat', 'stop_lon', 'stop_id']]
    mdf = pd.merge(wt_sub, fs_sub, on='stop_id', how='left')
    return mdf[~mdf.isnull()]


def populate_graph(G: nx.MultiDiGraph,
                   name: str,
                   feed: ptg.gtfs.feed,
                   wait_times_by_stop: pd.DataFrame,
                   summary_edge_costs: pd.DataFrame,
                   connection_threshold: Union[int, float],
                   walk_speed_kmph: float=4.5):
    # As we convert stop ids to actual nodes, let's keep track of those names
    # here so that we can reference them when we add connector edges across
    # the various feeds loaded into the graph
    sid_lookup = {}

    # Generate a merge of the wait time data and the feed stops data that will
    # be used for both the addition of new stop nodes and the addition of
    # cross feed edges later on (that join one feeds stops to the other if
    # they are within the connection threshold).
    stops_df = _merge_stop_waits_and_attributes(wait_times_by_stop, feed.stops)

    for i, row in stops_df.iterrows():
        sid = str(row.stop_id)
        full_sid = nameify_stop_id(name, sid)

        # Add to the lookup crosswalk dictionary
        sid_lookup[sid] = full_sid

        G.add_node(full_sid,
                   boarding_cost=row.avg_cost,
                   y=row.stop_lat,
                   x=row.stop_lon)

    for i, row in summary_edge_costs.iterrows():
        sid_fr = nameify_stop_id(name, row.from_stop_id)
        sid_to = nameify_stop_id(name, row.to_stop_id)
        G.add_edge(sid_fr,
                   sid_to,
                   length=row.edge_cost)

    # Generate cross feed edge values
    cross_feed_edges = generate_cross_feed_edges(G,
                                                 stops_df,
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
        kmph = (d / 1000) / walk_speed_kmph

        # Convert to seconds
        in_seconds = kmph * 60 * 60

        G.add_edge(full_sid, to, length=in_seconds)

    return G
