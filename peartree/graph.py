from typing import Dict, List, Union

import networkx as nx
import pandas as pd
import partridge as ptg
from fiona import crs
from shapely.geometry import shape

from .settings import WGS84
from .summarizer import (generate_edge_and_wait_values,
                         generate_summary_edge_costs,
                         generate_summary_wait_times)
from .synthetic import (generate_edges_df, generate_meter_projected_chunks,
                        generate_nodes_df, generate_stop_ids,
                        generate_stop_points)
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
                                    target_time_end: int,
                                    fallback_stop_cost: float,
                                    interpolate_times: bool,
                                    use_multiprocessing: bool):
    (all_edge_costs,
     all_wait_times) = generate_edge_and_wait_values(feed,
                                                     target_time_start,
                                                     target_time_end,
                                                     interpolate_times,
                                                     use_multiprocessing)

    # Same sanity checks on the output before we continue
    _verify_outputs(all_edge_costs, all_wait_times)

    summary_edge_costs = generate_summary_edge_costs(all_edge_costs)
    wait_times_by_stop = generate_summary_wait_times(all_wait_times,
                                                     fallback_stop_cost)

    return (summary_edge_costs, wait_times_by_stop)


def generate_cross_feed_edges(G: nx.MultiDiGraph,
                              name: str,
                              stops_df: pd.DataFrame,
                              exempt_nodes: List[str],
                              connection_threshold: float) -> pd.DataFrame:
    # Terminate this process early if the graph is empty
    if (G.number_of_nodes() == 0):
        return pd.DataFrame({'stop_id': [],
                             'to_nodes': [],
                             'edge_costs': []})

    # First, we need a DataFrame representation of the nodes in the graph
    node_df = generate_graph_node_dataframe(G)

    # Remove all nodes that are part of the new additions to the graph
    if len(exempt_nodes) > 0:
        node_df = node_df[~node_df.index.isin(exempt_nodes)]

    stop_ids = []
    to_nodes = []
    edge_costs = []

    # TODO: Repeating this in populate_graph as well, there may
    #       be a way to condense these two steps well
    for i, row in stops_df.iterrows():
        sid = str(row.stop_id)
        full_sid = nameify_stop_id(name, sid)

        # Ensure that each value is typed correctly prior to being
        # fed into the nearest node method
        lat = float(row.stop_lat)
        lon = float(row.stop_lon)
        point = (lat, lon)
        nearest_nodes = get_nearest_nodes(node_df,
                                          point,
                                          connection_threshold,
                                          exempt_id=full_sid)

        # Iterate through series results and add to output
        for node_id, dist_val in nearest_nodes.iteritems():
            stop_ids.append(sid)
            to_nodes.append(node_id)
            edge_costs.append(dist_val)

    return pd.DataFrame({'stop_id': stop_ids,
                         'to_node': to_nodes,
                         'distance': edge_costs})


def _verify_outputs(all_edge_costs: pd.DataFrame,
                    all_wait_times: pd.DataFrame) -> None:
    # Handle if there are no valid edges returned (or wait times)
    if all_edge_costs is None or len(all_edge_costs) == 0:
        raise InsufficientSummaryResults('The target time frame returned no '
                                         'valid edge costs from feed object.')
    if all_wait_times is None or len(all_wait_times) == 0:
        raise InsufficientSummaryResults('The target time frame returned no '
                                         'valid wait times from feed object.')


def _merge_stop_waits_and_attributes(wait_times_by_stop: pd.DataFrame,
                                     feed_stops: pd.DataFrame) -> pd.DataFrame:
    wt_sub = wait_times_by_stop[['avg_cost', 'stop_id']]
    fs_sub = feed_stops[['stop_lat', 'stop_lon', 'stop_id']]
    mdf = pd.merge(wt_sub, fs_sub, on='stop_id', how='left')
    return mdf[~mdf.isnull()]


def _add_cross_feed_edges(G: nx.MultiDiGraph,
                          sid_lookup: Dict[str, str],
                          cross_feed_edges: pd.DataFrame,
                          walk_speed_kmph: float) -> nx.MultiDiGraph:
    # Add the cross feed edge connectors to the graph to
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

        # And then add it to the graph
        G.add_edge(full_sid, to, length=in_seconds, mode='walk')


def _add_nodes_and_edges(G: nx.MultiDiGraph,
                         name: str,
                         stops_df: pd.DataFrame,
                         summary_edge_costs: pd.DataFrame,
                         bidirectional: bool=False) -> Dict[str, str]:
    # As we convert stop ids to actual nodes, let's keep track of those names
    # here so that we can reference them when we add connector edges across
    # the various feeds loaded into the graph
    sid_lookup = {}

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
        G.add_edge(sid_fr, sid_to, length=row.edge_cost, mode='transit')

        # If want to add both directions in this step, we can
        # by reversing the to/from node order on edge
        if bidirectional:
            G.add_edge(sid_to, sid_fr, length=row.edge_cost, mode='transit')

    return sid_lookup


def populate_graph(G: nx.MultiDiGraph,
                   name: str,
                   feed: ptg.gtfs.feed,
                   wait_times_by_stop: pd.DataFrame,
                   summary_edge_costs: pd.DataFrame,
                   connection_threshold: Union[int, float],
                   walk_speed_kmph: float=4.5,
                   impute_walk_transfers: bool=True):
    # Generate a merge of the wait time data and the feed stops data that will
    # be used for both the addition of new stop nodes and the addition of
    # cross feed edges later on (that join one feeds stops to the other if
    # they are within the connection threshold).
    stops_df = _merge_stop_waits_and_attributes(wait_times_by_stop, feed.stops)

    # Mutates the G network object
    sid_lookup = _add_nodes_and_edges(G, name, stops_df, summary_edge_costs)

    # Default to exempt new edges created, unless imputing internal
    # walk transfers is requested as well
    exempt_nodes = sid_lookup.values()
    if impute_walk_transfers:
        # In which case, we do not have any exempt nodes
        exempt_nodes = []
    cross_feed_edges = generate_cross_feed_edges(G, name, stops_df,
                                                 exempt_nodes,
                                                 connection_threshold)

    # Mutates the G network object
    _add_cross_feed_edges(G, sid_lookup, cross_feed_edges, walk_speed_kmph)

    return G


def _validate_feature_properties(props: Dict) -> Dict:
    fresh_props = {}
    fresh_props['headway'] = float(props['headway'])
    fresh_props['average_speed'] = float(props['average_speed'])

    # Check if want to do bidirectional (optional)
    fresh_props['bidirectional'] = props.get('bidirectional', False)

    # If the user supplied custom stops, let's try and use them
    # otherwise this attribute will be passed as a Nonetype
    fresh_props['custom_stops'] = props.get('stops', None)

    # Similarly, if there are stops distance instead, then we can use that
    fresh_props['stop_dist'] = props.get('stop_distance_distribution', None)

    # Sanity check; if both custom stops and stops distance are None
    # then we cannot proceed
    no_stops = fresh_props['custom_stops'] is None
    no_dist = fresh_props['stop_dist'] is None
    if no_stops and no_dist:
        raise ValueError('Synthetic network addition must have either '
                         'custom stops or stops distance default set.')

    return fresh_props


def make_synthetic_system_network(
        G: nx.MultiDiGraph,
        name: str,
        reference_geojson: Dict,
        connection_threshold: Union[int, float],
        walk_speed_kmph: float=4.5,
        impute_walk_transfers: bool=True):
    # Same as populate_graph, we use this dict to monitor the stop ids
    # that are created
    sid_lookup = {}
    all_nodes = None
    for feat in reference_geojson['features']:
        # Pull out required properties
        props = _validate_feature_properties(feat['properties'])
        headway = props['headway']
        avg_speed = props['average_speed']
        stop_dist = props['stop_dist']
        custom_stops = props['custom_stops']

        # We require this GeoJSON coordinate component to be valid format
        route_path = shape(feat['geometry'])

        # Generate reference geometry data, note (and this is confusing) but
        # chunks is in meter projection and all_pts is in web mercator
        # this is because we only need (from chunks) the length value
        # and do not actually preserve the geometry beyond these operations
        chunks = generate_meter_projected_chunks(route_path,
                                                 custom_stops,
                                                 stop_dist)
        all_pts = generate_stop_points(chunks)

        # Give each stop a unique id
        stop_ids = generate_stop_ids(len(all_pts))

        # Produce graph components
        nodes = generate_nodes_df(stop_ids, all_pts, headway)
        edges = generate_edges_df(stop_ids, chunks, avg_speed)

        # Mutates the G network object
        sid_lookup_sub = _add_nodes_and_edges(
            G, name, nodes, edges, props['bidirectional'])

        # Update the parent sid with new values
        for key, val in sid_lookup_sub.items():
            sid_lookup[key] = val

        # Then add to the running tally of nodes
        if all_nodes is None:
            all_nodes = nodes.copy()
        else:
            all_nodes = all_nodes.append(nodes)

    # Default to exempt new edges created, unless imputing internal
    # walk transfers is requested as well
    exempt_nodes = sid_lookup.values()
    if impute_walk_transfers:
        # In which case, we do not have any exempt nodes
        exempt_nodes = []
    cross_feed_edges = generate_cross_feed_edges(G, name, all_nodes,
                                                 exempt_nodes,
                                                 connection_threshold)
    # Mutates the G network object
    _add_cross_feed_edges(G, sid_lookup, cross_feed_edges, walk_speed_kmph)

    return G
