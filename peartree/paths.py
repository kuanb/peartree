import string
import random

import partridge as ptg

from .utilities import log


class InvalidGTFS(Exception):
    # Let's have a custom exception for when we read in GTFS files
    pass


def generate_random_name(N=5):
    choices = (string.ascii_uppercase + string.digits)
    return ''.join(random.SystemRandom().choice(choices) for _ in range(N))


def get_representative_feed(file_loc: str,
                            day_type: str='busiest',
                            name: str=None):
    # Generate a random name for name if it is None
    if not name:
        name = generate_random_name()
    
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
        raise NotImplementedError('Unsupported day type option supplied.')

    log(f'Selected_date: {selected_date}')
    log(f'Number of trips on that date: {trip_count}')

    all_service_ids = '\n\t'.join(service_ids_by_date[selected_date])
    log(f'\nAll related service IDs: \n\t{all_service_ids}')

    feed_query = {'trips.txt': {'service_id': service_ids_by_date[selected_date]}}
    return ptg.feed(file_loc, view=feed_query)


def load_feed_as_graph(feed):
    pass

