import datetime as dt
import logging as lg
import os
import sys
import unicodedata

from . import settings


def get_logger(level=None,
               name=None,
               filename=None):
    # Taken from OSMnx's utils.py file, see log comments
    # for link to version from which these methods were taken

    if level is None:
        level = settings.log_level
    if name is None:
        name = settings.log_name
    if filename is None:
        filename = settings.log_filename

    logger = lg.getLogger(name)

    # if a logger with this name is not already set up
    if not getattr(logger, 'handler_set', None):

        # get today's date and construct a log filename
        todays_date = dt.datetime.today().strftime('%Y_%m_%d')
        log_filename = '{}/{}_{}.log'.format(
            settings.logs_folder, filename, todays_date)

        # if the logs folder does not already exist, create it
        if not os.path.exists(settings.logs_folder):
            os.makedirs(settings.logs_folder)

        # create file handler and log formatter and set them up
        handler = lg.FileHandler(log_filename, encoding='utf-8')
        formatter = lg.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.handler_set = True

    return logger


def config(log_console=settings.log_console):
    # Taken from OSMnx's utils.py file, see log comments
    # for link to version from which these methods were taken

    # Set each global variable to the passed-in parameter value
    settings.log_console = log_console

    # if logging is turned on, log that we are configured
    if settings.log_file or settings.log_console:
        log('Configured osmnx')


def make_str(value):
    # This method should I ever want to support Python 2.x
    try:
        # For python 2.x compatibility, use unicode
        return unicode(value)
    except NameError:
        # Python 3.x has no unicode type, so if error, use str type
        return str(value)


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
        str_msg = make_str(message)
        normalized = unicodedata.normalize('NFKD', str_msg)
        encoded = normalized.encode('ascii', errors='replace')
        decoded = encoded.decode()
        print(decoded)
        sys.stdout = standard_out
