from typing import Dict

import networkx as nx
import partridge as ptg

from .graph import (generate_empty_md_graph, generate_summary_graph_elements,
                    make_synthetic_system_network, populate_graph)
from .toolkit import generate_random_name
from .utilities import log

FALLBACK_STOP_COST_DEFAULT = (30 * 60)  # 30 minutes, converted to seconds


class InvalidGTFS(Exception):
    # Let's have a custom exception for when we read in GTFS files
    pass


class InvalidTimeBracket(Exception):
    pass


def get_representative_feed(file_loc: str,
                            day_type: str='busiest'):
    # Extract service ids and then trip counts by those dates
    service_ids_by_date = ptg.read_service_ids_by_date(file_loc)
    trip_counts_by_date = ptg.read_trip_counts_by_date(file_loc)

    # Make sure we have some valid values returned in trips
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
    return ptg.feed(file_loc, view=feed_query)


def load_feed_as_graph(feed: ptg.gtfs.feed,
                       start_time: int,
                       end_time: int,
                       name: str=None,
                       existing_graph: nx.MultiDiGraph=None,
                       connection_threshold: float=50.0,
                       walk_speed_kmph: float=4.5,
                       fallback_stop_cost: bool=FALLBACK_STOP_COST_DEFAULT,
                       interpolate_times: bool=True,
                       impute_walk_transfers: bool=False,
                       use_multiprocessing: bool=False):
    """
    Convert a feed object into a NetworkX Graph, connect to an existing
    NetworkX graph if one is supplied

    Parameters
    ----------
    feed : partridge.feed
        A feed object from Partridge holding a representation of the
        desired schedule ids and their releated scheudule data from an
        operator GTFS
    start_time : int
        Represented in seconds after midnight; indicates the start time
        with which to take the subset of the target feed schedule
        to be used to measure impedance between stops along
        the route, as well as cost (wait time) to board at each stop
    end_time : int
        Represented in seconds after midnight; indicates the end time
        with which to take the subset of the target feed schedule
        to be used to measure impedance between stops along
        the route, as well as cost (wait time) to board at each stop
    name : str
        Name of the operator, which is used to create a unique ID for each
        of the stops, routes, etc. in the feed being supplied
    existing_graph : networkx.Graph
        An existing graph containing other operator or schedule data
    connection_threshold : float
        Treshold by which to create a connection with an existing stop
        in the existing_graph graph, measured in meters
    walk_speed_kmph : float
        Walk speed in km/h, that is used to determine the cost in time when
        walking between two nodes that get an internal connection created
    fallback_stop_cost: bool
        Cost in seconds to board a line at a stop if no other data is able
        to be calculated from schedule data for that stop to determine
        what wait time is. Example of this situation would be when
        there is only one scheduled stop time found for the stop id.
    interpolate_times : bool
        A boolean flag to indicate whether or not to infill intermediary stops
        that do not have all intermediary stop arrival times specified in the
        GTFS schedule.
    impute_walk_transfers : bool
        A flag to indicate whether to add in walk connections between nodes
        that are close enough, as measured using connection_trheshold
    use_multiprocessing: bool
        A flag to indicate whether or not to leverage multiprocessing where
        available to attempt to speed up trivially parallelizable operations.

    Returns
    -------
    G
        networkx.Graph, the loaded, combined representation of the schedule
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
        impute_walk_transfers: bool=True):
    """
    Convert a formatter transit FeatureCollection into a directed network graph.

    Utilizing a correctly formatted transit FeatureCollection, generate a
    directed networ graph (or add to an existing one), based off of features
    included in the reference_geojson parameter.
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
    else:
        G = generate_empty_md_graph(name)

    # TODO: Refactor reference_geojson to become a class that includes
    #       validation on instantiation

    return make_synthetic_system_network(
        G,
        name,
        reference_geojson,
        connection_threshold,
        walk_speed_kmph,
        impute_walk_transfers)
