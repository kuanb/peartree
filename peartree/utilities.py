import logging as lg

import partridge as ptg


def config(log_console=settings.log_console):
    # Taken from OSMnx's utils.py file, see log comments
    # for link to version from which these methods were taken
    
    # Set each global variable to the passed-in parameter value
    settings.log_console = log_console

    # if logging is turned on, log that we are configured
    if settings.log_file or settings.log_console:
        log('Configured osmnx')


def log(message: str, level=None, name=None, filename=None):
    # Same function, taken from OSMnx's log utility
    # Link: https://github.com/gboeing/osmnx/blob/
    #       0f284ae78ccbf732f5550f96d0deebe287dab115/osmnx/utils.py#L95

    if level is None:
        level = settings.log_level
    if name is None:
        name = settings.log_name
    if filename is None:
        filename = settings.log_filename

    # If logging to file is turned on
    if settings.log_file:
        # Get the current logger (or create a new one, if none), then log
        # message at requested level
        logger = get_logger(level=level, name=name, filename=filename)
        if level == lg.DEBUG:
            logger.debug(message)
        elif level == lg.INFO:
            logger.info(message)
        elif level == lg.WARNING:
            logger.warning(message)
        elif level == lg.ERROR:
            logger.error(message)

    # If logging to console is turned on, convert message to ascii and print to
    # the console
    if settings.log_console:
        # Capture current stdout, then switch it to the console, print the
        # message, then switch back to what had been the stdout. this prevents
        # logging to notebook - instead, it goes to console
        standard_out = sys.stdout
        sys.stdout = sys.__stdout__

        # Convert message to ascii for console display so it doesn't break
        # windows terminals
        message = unicodedata.normalize('NFKD', make_str(message)).encode('ascii', errors='replace').decode()
        print(message)
        sys.stdout = standard_out


def get_representative_feed(day_type: str, file_loc: str, name: str=None):
    """
    Get a subset of the feed by a specific type of day (e.g. busiest).

    Parameters
    ----------
    day_type : string
        a supported categorical that will indicate what kind day you want from the schedule
        as determined by schedule frequency
    file_loc : string
        location path to the zip file of the gtfs
    name : string
        the name of the feed being produced (if none, will auto-generate)

    Returns
    -------
    feed : a partridge object representing a subset of a transit feed for a given schedule id
    """



    # Generate a random name for name if it is None
    if not name:
        name = _generate_random_name()
    
    service_ids_by_date = ptg.read_service_ids_by_date(gtfs_loc)
    trip_counts_by_date = ptg.read_trip_counts_by_date(gtfs_loc)

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
    return ptg.feed(gtfs_loc, view=feed_query)
