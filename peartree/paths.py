import string
from typing import Dict
import random

import partridge as ptg

from .graph import (generate_empty_md_graph,
                    generate_summary_graph_elements,
                    populate_graph)
from .settings import WGS84
from .utilities import log


class InvalidGTFS(Exception):
    # Let's have a custom exception for when we read in GTFS files
    pass


class InvalidTimeBracket(Exception):
    pass


def _generate_random_name(N: int=5):
    choices = (string.ascii_uppercase + string.digits)
    return ''.join(random.SystemRandom().choice(choices) for _ in range(N))


def get_representative_feed(file_loc: str,
                            day_type: str='busiest'):
    
    service_ids_by_date = ptg.read_service_ids_by_date(file_loc)
    trip_counts_by_date = ptg.read_trip_counts_by_date(file_loc)

    # Make sure we have some valid values returned in trips
    if not len(trip_counts_by_date.items()):
        # Otherwise, error out
        raise InvalidGTFS('No valid trip counts by date were identified in GTFS.')

    # At this point, different methods can be implemented to help select how
    # to pick which date/schedule id to use
    if day_type == 'busiest':
        (selected_date,
         trip_count) = max(trip_counts_by_date.items(), key=lambda p: p[1])
    else:
        raise NotImplementedError('Unsupported day type string supplied.')

    log(f'Selected_date: {selected_date}')
    log(f'Number of trips on that date: {trip_count}')

    all_service_ids = '\n\t'.join(service_ids_by_date[selected_date])
    log(f'\nAll related service IDs: \n\t{all_service_ids}')

    feed_query = {'trips.txt': {'service_id': service_ids_by_date[selected_date]}}
    return ptg.feed(file_loc, view=feed_query)


def load_feed_as_graph(feed: ptg.gtfs.feed,
                       start_time: int,
                       end_time: int,
                       name: str=None,
                       existing_graph: nx.MultiDiGraph=None):
    # Generate a random name for name if it is None
    if not name:
        name = _generate_random_name()

    # Some sanity checking, to make sure only positive values are provided
    if (start_time < 0) or (end_time < 0):
        raise InvalidTimeBracket('Invalid start or end target times provided.')

    if end_time < start_time:
        raise InvalidTimeBracket('Invalid ordering: Start time is greater than end time.')

    (summary_edge_costs,
     wait_times_by_stop) = generate_summary_graph_elements(feed, start_time, end_time)

    # This is a flag used to check if we need to run any additional steps
    # after the feed is returned to ensure that new nodes and edge can connect
    # with existing ones (if they exist/a graph is passed in)
    existing_graph_supplied = bool(existing_graph)

    # G is either a new MultiDiGraph or one pass from before
    if existing_graph_supplied:
        # TODO: If passed from before we should run some checks to ensure it is valid
        #       as well as set a flag to create join points with other feeds so that
        #       they can be linked when the next is added.
        G = existing_graph
    else:
        G = generate_empty_md_graph(name)
    
    return populate_graph(G, name, feed, wait_times_by_stop, summary_edge_costs)
