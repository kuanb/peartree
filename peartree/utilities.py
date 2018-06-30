import datetime as dt
import logging as lg
import os
import unicodedata
import zipfile
from tempfile import TemporaryDirectory

import networkx as nx
import pandas as pd

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
        log('Configured peartree')


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
        # Convert message to ascii for console display so it doesn't break
        # windows terminals
        str_msg = make_str(message)
        normalized = unicodedata.normalize('NFKD', str_msg)
        encoded = normalized.encode('ascii', errors='replace')
        decoded = encoded.decode()
        print(decoded)


def save_graph_to_zip(G: nx.MultiDiGraph, path: str='peartree_graph.zip'):
    # Create a temporary workspace to save csvs to
    with TemporaryDirectory() as dirpath:
        # Extract the nodes from the graph, with attributes
        nodes_rows = []
        for node_id, node in G.nodes(data=True):
            nodes_rows.append({
                'id': node_id,
                'boarding_cost': node['boarding_cost'],
                'x': node['x'],
                'y': node['y']})

        # Roll up node rows into a DataFrame
        nodes_df = pd.DataFrame(nodes_rows)

        # Make sure that the column order is consistent
        nodes_df = nodes_df[['id', 'boarding_cost', 'x', 'y']]

        # Save that DataFrame to a csv
        nodes_fpath = '{}/nodes.csv'.format(dirpath)
        nodes_df.to_csv(nodes_fpath)

        # Extract the nodes from the graph, with attributes
        edges_rows = []
        for from_id, to_id, edge in G.edges(data=True):
            edges_rows.append({
                'from': from_id,
                'to': to_id,
                'length': edge['length'],
                'mode': edge['mode']})

        # Roll up node rows into a DataFrame
        edges_df = pd.DataFrame(edges_rows)

        # Make sure that the column order is consistent
        edges_df = edges_df[['from', 'to', 'length', 'mode']]

        # Save that DataFrame to a csv
        edges_fpath = '{}/edges.csv'.format(dirpath)
        edges_df.to_csv(edges_fpath)

        # Produce an output location handler
        zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)

        # Write all components of temp dir to that handler
        for fpath, fname in [(nodes_fpath, 'nodes.csv'),
                             (edges_fpath, 'edges.csv')]:
            zipf.write(fpath, fname)

        # Can now close handler
        zipf.close()


def graph_from_zip(path_to_zip_file: str) -> nx.MultiDiGraph:
    with TemporaryDirectory() as dirpath:
        zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
        zip_ref.extractall(dirpath)
        zip_ref.close()

        nodes_df = pd.read_csv('{}/nodes.csv'.format(dirpath))
        edges_df = pd.read_csv('{}/edges.csv'.format(dirpath))

    # Create an empty MDG
    G_from_zip = nx.MultiDiGraph()

    # First, work through the nodes and add them all
    for i, node in nodes_df.iterrows():
        G_from_zip.add_node(node['id'],
                            boarding_cost=node['boarding_cost'],
                            x=node['x'],
                            y=node['y'])

    # Second, work through the edges dataframe
    for i, edge in edges_df.iterrows():
        G_from_zip.add_edge(edge['from'],
                            edge['to'],
                            length=edge['length'],
                            mode=edge['mode'])

    return G_from_zip
