from typing import Any, Dict, List

import networkx as nx
import numpy as np
import partridge as ptg

from .graph import (generate_empty_md_graph, generate_summary_graph_elements,
                    make_synthetic_system_network, populate_graph)
from .synthetic import SyntheticTransitNetwork
from .toolkit import generate_random_name
from .utilities import generate_nodes_gdf_from_graph, log

FALLBACK_STOP_COST_DEFAULT = (30 * 60)  # 30 minutes, converted to seconds


class InvalidGTFS(Exception):
    # Let's have a custom exception for when we read in GTFS files
    pass


class InvalidTimeBracket(Exception):
    pass


def _calculate_means_default(
        target_time_start: float,
        target_time_end: float,
        arrival_times: List) -> float:
    # This is the default method that is provided to the load feed operation
    # and applied to the observed arrival times at a given stop. From this
    # array of arrival times, the average delay between stops is calcualted
    if len(arrival_times) < 2:
        return np.nan

    # Make sure that values are in ascending order (also converts to list)
    arrival_times = np.array(arrival_times)
    arrival_times.sort()

    # Recast as numpy array
    first = arrival_times[1:]
    second = arrival_times[:-1]
    wait_seconds = list(first - second)

    # Recast arrival times as just a python list
    arrival_times = list(arrival_times)

    # Also ensure that both the first and last trip include context
    # framed by the evaluation time period
    from_start_time_to_first_arrival = arrival_times[0] - target_time_start
    wait_seconds.append(from_start_time_to_first_arrival)

    from_last_arrival_to_end_time = target_time_end - arrival_times[-1]
    wait_seconds.append(from_last_arrival_to_end_time)

    # Note: Can implement something more substantial here that takes into
    #       account divergent/erratic performance or intentional timing
    #       clusters that are not evenly dispersed in a custom method that
    #       would replace this default method
    na = np.array(wait_seconds)

    # Prune 0-second delays as these excessively reduce wait-time estimates
    na_no_zeroes = na[na > 0]

    # Naive implementation: halve the headway to get average wait time
    average_wait = na_no_zeroes.mean() / 2
    return average_wait


def get_representative_feed(file_loc: str,
                            day_type: str='busiest') -> ptg.gtfs.Feed:
    """
    Given a filepath, extract a partridge feed object, holding a \
    representative set of schedule patterns, extracted from the GTFS zip \
    file, as a set of pandas DataFrames.

    Parameters
    ----------
    file_loc : str
        The location (filepath) of the GTFS zip file.
    day_type : str
        The name of the type of representative feed desired. Currently, only \
        one type is supported, busiest. This extracts the schedule pattern \
        for a day that has the most service on it. This is determined by the \
        day with the most trips on it.

    Returns
    -------
    feed : ptg.gtfs.Feed
        A partridge feed object, holding related schedule information as \
        pandas DataFrames for the busiest day in the available schedule.
    """

    # Extract service ids and then trip counts by those dates
    try:
        service_ids_by_date = ptg.read_service_ids_by_date(file_loc)
        trip_counts_by_date = ptg.read_trip_counts_by_date(file_loc)

    # Raised by partridge if no valid dates returned
    except AssertionError:
        # Make sure we have some valid values returned in trips
        raise InvalidGTFS('No valid trip counts by date '
                          'were identified in GTFS.')

    # TODO: Due to partridge's assertion error being raised, this
    #       check may no longer be needed.
    if not len(trip_counts_by_date.items()):
        # Otherwise, error out
        raise InvalidGTFS('No valid trip counts by date '
                          'were identified in GTFS.')

    # At this point, different methods can be implemented to help select how
    # to pick which date/schedule id to use
    if day_type == 'busiest':
        # Choose the service id that has the most trips associated with it
        (selected_date,
         trip_count) = max(trip_counts_by_date.items(), key=lambda p: p[1])
    else:
        raise NotImplementedError('Unsupported day type string supplied.')

    log('Selected_date: {}'.format(selected_date))
    log('Number of trips on that date: {}'.format(trip_count))

    all_service_ids = '\n\t'.join(service_ids_by_date[selected_date])
    log('\nAll related service IDs: \n\t{}'.format(all_service_ids))

    sub = service_ids_by_date[selected_date]
    feed_query = {'trips.txt': {'service_id': sub}}
    return ptg.load_feed(file_loc, view=feed_query)


def load_feed_as_graph(feed: ptg.gtfs.Feed,
                       start_time: int,
                       end_time: int,
                       name: str=None,
                       existing_graph: nx.MultiDiGraph=None,
                       connection_threshold: float=50.0,
                       walk_speed_kmph: float=4.5,
                       stop_cost_method: Any=_calculate_means_default,
                       fallback_stop_cost: bool=FALLBACK_STOP_COST_DEFAULT,
                       interpolate_times: bool=True,
                       impute_walk_transfers: bool=False,
                       use_multiprocessing: bool=False):
    """
    Convert a feed object into a NetworkX Graph, or connect to an existing \
    NetworkX graph if one is supplied.

    Parameters
    ----------
    feed : ptg.gtfs.Feed
        A feed object from Partridge holding a representation of the \
        desired schedule ids and their releated scheudule data from an \
        operator GTFS
    start_time : int
        Represented in seconds after midnight; indicates the start time \
        with which to take the subset of the target feed schedule \
        to be used to measure impedance between stops along \
        the route, as well as cost (wait time) to board at each stop
    end_time : int
        Represented in seconds after midnight; indicates the end time \
        with which to take the subset of the target feed schedule \
        to be used to measure impedance between stops along \
        the route, as well as cost (wait time) to board at each stop
    name : str
        Name of the operator, which is used to create a unique ID for each \
        of the stops, routes, etc. in the feed being supplied
    existing_graph : networkx.Graph
        An existing graph containing other operator or schedule data
    connection_threshold : float
        Treshold by which to create a connection with an existing stop \
        in the existing_graph graph, measured in meters
    walk_speed_kmph : float
        Walk speed in km/h, that is used to determine the cost in time when \
        walking between two nodes that get an internal connection created
    stop_cost_method : Any
        A method is passed in here that handles an arrival time numpy array \
        and, from that array, calcualtes a representative average wait time \
        value, in seconds, for that stop.
    fallback_stop_cost: bool
        Cost in seconds to board a line at a stop if no other data is able \
        to be calculated from schedule data for that stop to determine \
        what wait time is. Example of this situation would be when \
        there is only one scheduled stop time found for the stop id.
    interpolate_times : bool
        A boolean flag to indicate whether or not to infill intermediary \
        stops that do not have all intermediary stop arrival times specified \
        in the GTFS schedule.
    impute_walk_transfers : bool
        A flag to indicate whether to add in walk connections between nodes \
        that are close enough, as measured using connection_trheshold
    use_multiprocessing: bool
        A flag to indicate whether or not to leverage multiprocessing where \
        available to attempt to speed up trivially parallelizable operations

    Returns
    -------
    G : nx.MultiDiGraph
        networkx.Graph, the loaded, combined representation of the schedule \
        data from the feed subset by the time parameters provided
    """

    # Generate a random name for name if it is None
    if not name:
        name = generate_random_name()

    # Some sanity checking, to make sure only positive values are provided
    if (start_time < 0) or (end_time < 0):
        raise InvalidTimeBracket('Invalid start or end target times provided.')

    if end_time <= start_time:
        raise InvalidTimeBracket('Invalid ordering: Start time '
                                 'is greater than end time.')

    (summary_edge_costs,
     wait_times_by_stop) = generate_summary_graph_elements(feed,
                                                           start_time,
                                                           end_time,
                                                           fallback_stop_cost,
                                                           interpolate_times,
                                                           stop_cost_method,
                                                           use_multiprocessing)

    # This is a flag used to check if we need to run any additional steps
    # after the feed is returned to ensure that new nodes and edge can connect
    # with existing ones (if they exist/a graph is passed in)
    existing_graph_supplied = bool(existing_graph)

    # G is either a new MultiDiGraph or one pass from before
    if existing_graph_supplied:
        # TODO: If passed from before we should run some checks to ensure
        #       it is valid as well as set a flag to create join points with
        #       other feeds so that they can be linked when the next is added.
        G = existing_graph
    else:
        G = generate_empty_md_graph(name)

    return populate_graph(G,
                          name,
                          feed,
                          wait_times_by_stop,
                          summary_edge_costs,
                          connection_threshold,
                          walk_speed_kmph,
                          impute_walk_transfers)


def load_synthetic_network_as_graph(
        reference_geojson: Dict,
        name: str=None,
        existing_graph: nx.MultiDiGraph=None,
        connection_threshold: float=50.0,
        walk_speed_kmph: float=4.5,
        impute_walk_transfers: bool=True,
        wait_time_cost_method: Any=lambda x: x / 2) -> nx.MultiDiGraph:
    """
    Convert formatted transit FeatureCollection into a directed network graph.

    Utilizing a correctly formatted transit FeatureCollection, generate a \
    directed networ graph (or add to an existing one), based off of features \
    included in the reference_geojson parameter.

    Parameters
    ———————
    reference_geojson : dict
        The TransitJSON; a specifically formatted GeoJSON
    name : str
        The name of the graph
    existing_graph : nx.MultiDiGraph
        An existing, populated transit NetworkX graph generated from peartree
    connection_threshold : float
        Distance in meters within which a nearby transit stops should be \
        deemed acceptably close for a walk transfer to be also added
    walk_speed_kmph : float
        Speed in kilometers per hour to be used as the reference walk speed \
        for calculating cost (impedance in time) of walk transfers
    impute_walk_transfers : bool
        A flag to indicate whether or not walk transfers should be calculated
    wait_time_cost_method: Any
        Function that, given a headway float value, produces a wait time value

    Returns
    ——
    G : nx.MultiDiGraph
        The muti-directed graph
    """

    # Generate a random name for name if it is None
    if not name:
        name = generate_random_name()

    # This is a flag used to check if we need to run any additional steps
    # after the feed is returned to ensure that new nodes and edge can connect
    # with existing ones (if they exist/a graph is passed in)
    existing_graph_supplied = bool(existing_graph)

    # G is either a new MultiDiGraph or one pass from before
    if existing_graph_supplied:
        # TODO: If passed from before we should run some checks to ensure
        #       it is valid as well as set a flag to create join points with
        #       other feeds so that they can be linked when the next is added.
        G = existing_graph
        existing_graph_nodes = generate_nodes_gdf_from_graph(
            G, to_epsg_crs=2163)
    else:
        G = generate_empty_md_graph(name)
        existing_graph_nodes = None

    # First, instantiate whole TransitJSON as a SyntheticTransitNetwork object;
    # will provide necessory validation prior to synthetic network construction
    as_synthetic_network = SyntheticTransitNetwork(
        reference_geojson,
        wait_time_cost_method,
        existing_graph_nodes)

    return make_synthetic_system_network(
        G,
        name,
        as_synthetic_network,
        connection_threshold,
        walk_speed_kmph,
        impute_walk_transfers)
