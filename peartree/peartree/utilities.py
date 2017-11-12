import logging as lg


def config(log_console=settings.log_console):
    # Taken from OSMnx's utils.py file, see log comments
    # for link to version from which these methods were taken
    
    # Set each global variable to the passed-in parameter value
    settings.log_console = log_console

    # if logging is turned on, log that we are configured
    if settings.log_file or settings.log_console:
        log('Configured osmnx')


def log(message, level=None, name=None, filename=None):
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
