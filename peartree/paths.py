from typing import Dict

import networkx as nx
import partridge as ptg

from .graph import (generate_empty_md_graph, generate_summary_graph_elements,
                    make_synthetic_system_network, populate_graph)
from .toolkit import generate_random_name
from .utilities import log


class InvalidGTFS(Exception):
    # Let's have a custom exception for when we read in GTFS files
    pass


class InvalidTimeBracket(Exception):
    pass


def get_representative_feed(file_loc: str,
                            day_type: str='busiest'):

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
                       interpolate_times: bool=True,
                       exempt_internal_edge_imputation: bool=True):
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
                                                           interpolate_times)

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
                          exempt_internal_edge_imputation)


def load_synthetic_network_as_graph(
        reference_geojson: Dict,
        name: str=None,
        existing_graph: nx.MultiDiGraph=None,
        connection_threshold: float=50.0,
        walk_speed_kmph: float=4.5,
        exempt_internal_edge_imputation: bool=False):

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

    return make_synthetic_system_network(
        G,
        name,
        reference_geojson,
        connection_threshold,
        walk_speed_kmph,
        exempt_internal_edge_imputation)
