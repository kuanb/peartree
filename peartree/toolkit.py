import numpy as np
import osmnx as ox
import pandas as pd


def great_circle_vec(lat1: float,
                     lng1: float,
                     lat2: float,
                     lng2: float,
                     earth_radius: int=6371009):
    # This method wraps the same in OSMnx, source:
    #   https://github.com/gboeing/osmnx/blob/
    #   b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L262
    return ox.utils.great_circle_vec(lat1, lng1, lat2, lng2, earth_radius)


def generate_graph_node_dataframe(G):
    # This method breaks out a portion of a similar method from
    # OSMnx's get_nearest_node; source:
    #   https://github.com/gboeing/osmnx/blob/
    #   b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L326
    if not G or (G.number_of_nodes() == 0):
        raise ValueError('G argument must be not be empty or '
                         'should contain at least one node')

    # Dump graph node coordinates array
    clist = [[node, data['x'], data['y']] for node, data in G.nodes(data=True)]
    coords = np.array(clist)

    # Then make into a Pandas DataFrame, with the node as index
    return pd.DataFrame(coords, columns=['node', 'x', 'y']).set_index('node')


def get_nearest_node(df_orig: pd.DataFrame,
                     point: tuple):
    # This method breaks out a portion of a similar method from
    # OSMnx's get_nearest_node; source:
    #   https://github.com/gboeing/osmnx/blob/
    #   b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L326

    # Make a copy of the DataFrame to prevent mutation outside of function
    df = df_orig.copy()

    # Add second column of reference points
    df['reference_y'] = point[0]
    df['reference_x'] = point[1]

    # TODO: OSMnx supports euclidean as well, for now we have a stumped
    #       version of this same function

    # Calculate distance vector using great circle distances (ie, for
    # spherical lat-long geometries)
    distances = great_circle_vec(lat1=df['reference_y'],
                                 lng1=df['reference_x'],
                                 lat2=df['y'],
                                 lng2=df['x'])

    # Calculate the final results to be returned
    nearest_node = int(distances.idxmin())
    nn_dist = distances.loc[nearest_node]

    # Returna as tuple
    return (nearest_node, nn_dist)
