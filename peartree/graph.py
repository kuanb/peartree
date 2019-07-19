from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import pandas as pd
import partridge as ptg
from fiona import crs

from .settings import WGS84
from .summarizer import (generate_edge_and_wait_values,
                         generate_summary_edge_costs,
                         generate_summary_wait_times,
                         get_modes_at_stops)
from .synthetic import SyntheticTransitNetwork
from .toolkit import generate_graph_node_dataframe, get_nearest_nodes


class InsufficientSummaryResults(Exception):
    pass


def generate_empty_md_graph(name: str,
                            init_crs: Dict=crs.from_epsg(WGS84)):
    """
    Generates an empty multi-directed graph.

    Parameters
    ———————
    name : str
        The name of the graph
    init_crs : dict
        The coordinate reference system to be assigned to the graph. Example \
        CRS would be `{'init': 'epsg:4326'}`

    Returns
    ——
    G : nx.MultiDiGraph
        The muti-directed graph
    """
    return nx.MultiDiGraph(name=name, crs=init_crs)


def nameify_stop_id(name, sid):
    name = str(name)
    sid = str(sid)
    return '{}_{}'.format(name, sid)


def generate_summary_graph_elements(
        feed: ptg.gtfs.Feed,
        target_time_start: float,
        target_time_end: float,
        fallback_stop_cost: float,
        interpolate_times: bool,
        stop_cost_method: Any,
        use_multiprocessing: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes in a feed with a series of settings and produces two results: \
    a table of edge costs and a table of stop (or nodes).

    Output tables represent the two primary components of a network graph: \
    the graph nodes and vertices. This method wraps the primary edge and wait \
    times generator with a few additional validation checks and summary steps.

    Parameters
    ———————
    Feed : ptg.gtfs.Feed
        A partridge feed object, holding related schedule information as pandas
        DataFrames for the busiest day in the available schedule.
    target_time_start : float
        Start time in hours (on a 24-hour range). This will be used to \
        determine what part of the available selected schedule to subselect. \
        For example, you could set 7.5, which would be read as 7.5 hours past \
        midnight and converted into seconds.
    target_time_end : float
        End time in hours (on a 24-hour range). This will be used to \
        determine what part of the available selected schedule to subselect. \
        For example, you could set 13.5, which would be read as 17.5 hours \
        past midnight and converted into seconds.
    fallback_stop_cost : float
        This is measured in seconds. So, if the value is 600, it will equate \
        to a 10 minute fallback cost. The fallback stop cost is used if \
        peartree is unable to determine a time between arrivals for a \
        specific route. This can happen when there is only one arrival in \
        the considered time frame. In such situations, the fallback value \
        is used in lieu of a calculated standard wait time for that stop node.
    interpolate_times : bool
        Flag to check if there are intermediary stop in the GTFS feed that do \
        not have specific schedule data associated with them and to impute \
        the approximate arrival and departure times at each of these stops.
    stop_cost_method : Any
        A method is passed in here that handles an arrival time numpy array
        and, from that array, calcualtes a representative average wait time
        value, in seconds, for that stop.
    use_multiprocessing : bool
        This is a flag to tell the peartree model whether to attempt to \
        parallelize the computing of route-stop average wait times. It can \
        be helpful in speeding up evaluation of larger GTFS feeds.


    Returns
    ——
    summary_edge_costs : pd.DataFrame
        The edge DataFrame holds the from and to node (stop) ids, as well \
        as the cost calculated for traversing that edge.
    wait_times_by_stop : pd.DataFrame
        The node, or stops, DataFrame, holds the the stop id and the average \
        boarding cost for that node (calculated based on the distribution of \
        arrival times) within the evaluation time range for the target subset \
        of the feed schedule.
    """

    (all_edge_costs,
     all_wait_times) = generate_edge_and_wait_values(
        feed,
        target_time_start,
        target_time_end,
        interpolate_times,
        stop_cost_method,
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
    """
    Generates a DataFrame containing a list of new node-pairs that should \
    have walk connection edges added between them, determined based on a \
    distance (proximity) threshold.

    Parameters
    ———————
    G : nx.MultiDiGraph
        The muti-directed graph
    name : str
        The name (based on the graph) to be used in creating unique ids
    stops_df : pd.DataFrame
        Dataframe holding all stops and associated stops columns from the \
        graph feed object
    exempt_nodes : List[str]
        An iterable of string objects representing the ids of nodes that \
        should not have connector edges created to or from them
    connection_threshold : float
        The distance (in meters) used to determine if a connector edge should \
        be created between two nodes

    Returns
    ——
    connector_edges_df : pd.DataFrame
        Dataframe holding all edges between existing nodes that were added, \
        with the attributes: from stop id, to stop id, and distance (in meters)
    """
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


def _add_modes_to_wait_times(
        feed: ptg.gtfs.Feed,
        wait_times_by_stop: pd.DataFrame):
    # Note: wait_times_by_stop dataframe has 2 columns when passed in:
    #           stop_id, avg_cost
    #       stops_modes_lookup will have 2 columns:
    #           stop_id, modes
    stops_modes_lookup = get_modes_at_stops(feed)

    # Merge the two onto one dataframe with a total of 3 columns:
    #       stop_id, avg_cost, modes
    return pd.merge(
        wait_times_by_stop,
        stops_modes_lookup,
        on='stop_id',
        how='left')


def _merge_stop_waits_and_attributes(wait_times_by_stop: pd.DataFrame,
                                     feed_stops: pd.DataFrame) -> pd.DataFrame:
    wt_sub = wait_times_by_stop[['avg_cost', 'modes', 'stop_id']]
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
                   modes=row.modes,
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
                   feed: ptg.gtfs.Feed,
                   wait_times_by_stop: pd.DataFrame,
                   summary_edge_costs: pd.DataFrame,
                   connection_threshold: Union[int, float],
                   walk_speed_kmph: float=4.5,
                   impute_walk_transfers: bool=True) -> nx.MultiDiGraph:
    """
    Takes an existing network graph object and adds all new nodes, transit \
    edges, and connector edges between existing and new nodes (stop ids).

    Parameters
    ———————
    G : nx.MultiDiGraph
        The muti-directed graph
    name : str
        The name (based on the graph) to be used in creating unique ids
    Feed : ptg.gtfs.Feed
        A partridge feed object, holding related schedule information as pandas
        DataFrames for the busiest day in the available schedule
    wait_times_by_stop : pd.DataFrame
        A DataFrame of nodes to be added with their associated boarding costs
    summary_edge_costs : pd.DataFrame
        A DataFrame of edge costs between node-pairs and the related costs \
        a mode type for each of them
    connection_threshold : Union[int, float]
        The distance (in meters) used to determine if a connector edge should \
        be created between two nodes
    walk_speed_kmph : float
        The distance (in meters) used to determine if a connector edge should \
        be created between two nodes
    impute_walk_transfers : bool
        Flag indicating whether or not to utilize the connection threshold \
        value to determine what new transfer edges could be added

    Returns
    ——
    G : nx.MultiDiGraph
        The muti-directed graph
    """
    wait_times_and_modes = _add_modes_to_wait_times(feed, wait_times_by_stop)

    # Generate a merge of the wait time data and the feed stops data that will
    # be used for both the addition of new stop nodes and the addition of
    # cross feed edges later on (that join one feeds stops to the other if
    # they are within the connection threshold).
    stops_df = _merge_stop_waits_and_attributes(
        wait_times_and_modes, feed.stops)

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
                                                 float(connection_threshold))

    # Mutates the G network object
    _add_cross_feed_edges(G, sid_lookup, cross_feed_edges, walk_speed_kmph)

    return G


def make_synthetic_system_network(
        G: nx.MultiDiGraph,
        name: str,
        synthetic_network: SyntheticTransitNetwork,
        connection_threshold: Union[int, float],
        walk_speed_kmph: float=4.5,
        impute_walk_transfers: bool=True) -> nx.MultiDiGraph:
    """
    Consume and validate an input TransitJSON for conversion to network \
    graph components. Add the resulting new network components to the \
    existing network graph provided as a positional argument.

    Parameters
    ———————
    G : nx.MultiDiGraph
        The muti-directed graph
    name : str
        The name (based on the graph) to be used in creating unique ids
    synthetic_network : SyntheticTransitNetwork
        An instantiated SyntheticTransitNetwork object containing all \
        features as individually instantiated and validated \
        SyntheticTransitLine objects
    connection_threshold : Union[int, float]
        The distance (in meters) used to determine if a connector edge should \
        be created between two nodes
    walk_speed_kmph : float
        The distance (in meters) used to determine if a connector edge should \
        be created between two nodes
    impute_walk_transfers : bool
        Flag indicating whether or not to utilize the connection threshold \
        value to determine what new transfer edges could be added

    Returns
    ——
    G : nx.MultiDiGraph
        The muti-directed graph
    """
    # Same as populate_graph, we use this dict to monitor the stop ids
    # that are created
    sid_lookup = {}
    all_nodes = None

    # Now, iterate through each line, extracting a single SyntheticTransitLine
    for line in synthetic_network.lines:
        nodes = line.nodes
        edges = line.edges

        # Mutates the G network object
        sid_lookup_sub = _add_nodes_and_edges(
            G, name, nodes, edges, line.bidirectional)

        # Update the parent sid with new values
        for key, val in sid_lookup_sub.items():
            sid_lookup[key] = val

        # TODO: Appending pandas DataFrames is a costly operation
        #       so identify what is needed from this DataFrame and extract
        #       those values instead
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
