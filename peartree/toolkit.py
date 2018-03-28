import random
import string
from typing import Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point


def great_circle_vec(lat1: float,
                     lng1: float,
                     lat2: float,
                     lng2: float,
                     earth_radius: int=6371009):
    # This method wraps the same in OSMnx, source:
    #   https://github.com/gboeing/osmnx/blob/
    #   b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L262
    return ox.utils.great_circle_vec(lat1, lng1, lat2, lng2, earth_radius)


def generate_random_name(N: int=5):
    choices = (string.ascii_uppercase + string.digits)
    return ''.join(random.SystemRandom().choice(choices) for _ in range(N))


def generate_graph_node_dataframe(G):
    # This method breaks out a portion of a similar method from
    # OSMnx's get_nearest_node; source:
    #   https://github.com/gboeing/osmnx/blob/
    #   b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L326
    if not G or (G.number_of_nodes() == 0):
        raise ValueError('G argument must be not be empty or '
                         'should contain at least one node')

    # Dump graph node coordinates array
    clist = []
    for node, data in G.nodes(data=True):
        # Ensure that each items is cast as the correct typegi
        x = float(data['x'])
        y = float(data['y'])
        clist.append([node, x, y])
    coords = np.array(clist)

    # Then make into a Pandas DataFrame, with the node as index (type string)
    df = pd.DataFrame(coords, columns=['node', 'x', 'y'])
    df['node'] = df['node'].astype(str)
    df = df.set_index('node')
    return df


def get_nearest_nodes(df_orig: pd.DataFrame,
                      point: Tuple[float, float],
                      connection_threshold: float,
                      exempt_id: str=None):
    # This method breaks out a portion of a similar method from
    # OSMnx's get_nearest_node; source:
    #   https://github.com/gboeing/osmnx/blob/
    #   b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L326

    # Make a copy of the DataFrame to prevent mutation outside of function
    df = df_orig.copy()

    if exempt_id is not None:
        df.index = df.index.astype(str)
        mask = ~(df.index == exempt_id)
        df = df[mask]

    # Add second column of reference points
    df['reference_y'] = point[0]
    df['reference_x'] = point[1]

    # TODO: OSMnx supports euclidean as well, for now we have a stumped
    #       version of this same function

    # Ensure each vectorized series is typed correctly
    ref_ys = df['reference_y'].astype(float)
    ref_xs = df['reference_x'].astype(float)
    ys = df['y'].astype(float)
    xs = df['x'].astype(float)

    # Calculate distance vector using great circle distances (ie, for
    # spherical lat-long geometries)
    distances = great_circle_vec(lat1=ref_ys,
                                 lng1=ref_xs,
                                 lat2=ys,
                                 lng2=xs)

    # Filter out nodes outside connection threshold
    mask = (distances < connection_threshold)
    nearest_nodes = distances[mask]

    # Return filtered series
    return nearest_nodes


def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.
    From: https://stackoverflow.com/questions/6518811/
          interpolate-nan-values-in-a-numpy-array#6518811

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return (np.isnan(y), lambda z: z.nonzero()[0])


def reproject(G: nx.MultiDiGraph, to_epsg: int=2163) -> nx.MultiDiGraph:
    # Avoid upstream mutation of the graph
    G = G.copy()

    # First extract current crs
    orig_crs = G.graph['crs']

    # And get the array of nodes from original graph
    ref_node_array = list(G.nodes(data=True))

    all_pts = []
    for i, node in ref_node_array:
        all_pts.append(Point(node['x'], node['y']))

    # Convert the collected nodes to GeoSeries
    gs = gpd.GeoSeries(all_pts)
    # And then reproject from original crs to new
    gs.crs = orig_crs
    gs = gs.to_crs(epsg=to_epsg)

    # Now iterate back through the reprojected points
    # and add each to it's respected node
    for (i, node), new_pt in zip(ref_node_array, gs):
        G.nodes[i]['x'] = new_pt.x
        G.nodes[i]['y'] = new_pt.y

    # Return the reprojected copy
    return G


def coalesce(G: nx.MultiDiGraph, resolution: float) -> nx.MultiDiGraph:
    # Avoid upstream mutation of the graph
    G = G.copy()

    # Extract all x, y values
    grouped = {}
    for i, node in G.nodes(data=True):
        x = (round(node['x'] / resolution) * resolution)
        y = (round(node['y'] / resolution) * resolution)

        # Build the dictionary as needed
        if x not in grouped:
            grouped[x] = {}
        if y not in grouped[x]:
            grouped[x][y] = []

        # Append each node under its approx. area grouping
        grouped[x][y].append(i)

    # Generate a series of reference dictionaries that allow us
    # to assign a new node name to each grouping of nodes
    counter = 0
    new_node_coords = {}
    lookup = {}


    # Populate the fresh reference dictionaries
    for x in grouped:
        for y in grouped[x]:
            new_node_name = '{}_{}'.format(G.name, counter)
            new_node_coords[new_node_name] = {'x': x, 'y': y}
            for n in grouped[x][y]:
                lookup[n] = new_node_name
            counter += 1

    # Recast the lookup crosswalk as a series for convenience
    reference = pd.Series(lookup)

    # Get the average boarding cost for each node grouping
    for nni in new_node_coords:
        boarding_costs = []
        g_nodes = reference.loc[reference == nni].index.values
        for i in g_nodes:
            bc = G.nodes[i]['boarding_cost']
            boarding_costs.append(bc)
        avg_bc = np.array(boarding_costs).mean()
        new_node_coords[nni]['boarding_cost'] = avg_bc

    # Create a list of replacement edges
    edges_to_add = []
    for n1, n2, edge in G.edges(data=True):
        edges_to_add.append((reference[n1], reference[n2], edge))

    # Add the new edges
    for n1, n2, edge in edges_to_add:
        # But avoid edges that now connect to the same node
        if not n1 == n2:
            G.add_edge(n1, n2, length=edge['length'], mode=edge['mode'])

    # Now we can remove all edges and nodes that predated the
    # coalescing operations
    for n in reference.index:
        # Note that this will also drop all edges
        G.remove_node(n)

    # Also make sure to update the new nodes with their summary
    # stats and locational data
    for i, node in new_node_coords.items():
        if i not in G.nodes():
            continue
        for key in node:
            G.nodes[i][key] = node[key]

    return G
