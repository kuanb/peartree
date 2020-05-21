import logging as lg
import random
import string
import warnings
from typing import List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .utilities import log


def great_circle_vec(lat1: float,
                     lng1: float,
                     lat2: float,
                     lng2: float,
                     earth_radius: float=6371009.0) -> float:
    """
    Vectorized function to calculate the great-circle distance between two
    points or between vectors of points.

    Please note that this method is copied from OSMnx method of the same name,
    which can be accessed here:
    https://github.com/gboeing/osmnx/blob/
    b32f8d333c6965a0d2f27c1f3224a29de2f08d55/osmnx/utils.py#L262

    Parameters
    ----------
    lat1 : float or array of float
    lng1 : float or array of float
    lat2 : float or array of float
    lng2 : float or array of float
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is
        meters)

    Returns
    -------
    distance : float
        distance or vector of distances from (lat1, lng1) to (lat2, lng2) in
        units of earth_radius
    """

    phi1 = np.deg2rad(90 - lat1)
    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2)
           + np.cos(phi1) * np.cos(phi2))

    # Ignore warnings during this calculation because numpy warns it cannot
    # calculate arccos for self-loops since u==v
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arc = np.arccos(cos)

    # Return distance in units of earth_radius
    distance = arc * earth_radius
    return distance


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

    return (np.isnan(y), lambda z: z.to_numpy().nonzero()[0])


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

    # Update the graph's coordinate reference
    G.graph['crs'] = {'init': 'epsg:{}'.format(to_epsg)}

    # Return the reprojected copy
    return G


def coalesce(
    G_orig: nx.MultiDiGraph,
    resolution: float,
    edge_summary_method=lambda x: x.max(),
    boarding_cost_summary_method=lambda x: x.mean(),
) -> nx.MultiDiGraph:
    # Note: Feature is experimental. For more details, see
    #       https://github.com/kuanb/peartree/issues/126
    warnings.warn((
        'coalesce method is experimental - method risks '
        'deformation of relative graph structure'))

    # Make sure our resolution satisfies basic requirement
    if resolution < 1:
        raise ValueError('Resolution parameters must be >= 1')

    # Avoid upstream mutation of the graph
    G = G_orig.copy()

    # Before we continue, attempt to simplfy the current network
    # such that we won't generate isolated nodes that become disconnected
    # from key coalesced nodes (because too many intermediary nodes)
    G = simplify_graph(G)

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

            # Pair each newly generate name to the original node id,
            # preserved from the original groupings resulting array
            for n in grouped[x][y]:
                lookup[n] = new_node_name

            # Update the counter so each new synthetic
            # node name will be different
            counter += 1

    # Recast the lookup crosswalk as a series for convenience
    reference = pd.Series(lookup)

    # Get the following attributes:
    #   1. average boarding cost for each node grouping
    #   2. modes associated with each node grouping
    for nni in new_node_coords:
        # Initialize an empty list
        boarding_costs = []
        all_modes_related = []

        # Get all original nodes that have been grouped
        g_nodes = reference.loc[reference == nni].index.values

        # Iterate through and add gather costs
        for i in g_nodes:
            specific_node = G.nodes[i]

            bc = specific_node['boarding_cost']
            boarding_costs.append(bc)

            this_nodes_modes = specific_node['modes']
            all_modes_related.extend(this_nodes_modes)

        # Calculate the summary boarding costs
        # and assign it to the new nodes objects
        new_node_coords[nni]['boarding_cost'] = (
            boarding_cost_summary_method(np.array(boarding_costs)))

        # Get all unique modes and assign it to the new nodes objects
        sorted_set_list = sorted(list(set(all_modes_related)))
        new_node_coords[nni]['modes'] = sorted_set_list

    # First step to creating a list of replacement edges
    replacement_edges_fr = []
    replacement_edges_to = []
    replacement_edges_len = []

    for n1, n2, edge in G.edges(data=True):
        # This will be used to parse out which edges to keep
        replacement_edges_fr.append(reference[n1])
        replacement_edges_to.append(reference[n2])
        replacement_edges_len.append(edge['length'])

    # This takes the resulting matrix and converts it to a pandas DataFrame
    edges_df = pd.DataFrame({
        'fr': replacement_edges_fr,
        'to': replacement_edges_to,
        'len': replacement_edges_len})

    # Next we group by the edge pattern (from -> to)
    grouped = edges_df.groupby(['fr', 'to'], sort=False)

    # With the resulting groupings, we extract values
    # TODO: Also group on modes
    processed_edge_costs = edge_summary_method(grouped['len'])

    # Second step; which uses results from edge_df grouping/parsing
    edges_to_add = []
    for n1, n2, edge in G.edges(data=True):
        # Get corresponding ids of new nodes (grid corners)
        ref_n1 = reference[n1]
        ref_n2 = reference[n2]

        # Retrieve pair value from previous grouping operation
        avg_length = processed_edge_costs.loc[ref_n1, ref_n2]
        edges_to_add.append((
            ref_n1,
            ref_n2,
            avg_length,
            edge['mode']))

    # Add the new edges to graph
    for n1, n2, length, mode in edges_to_add:
        # Only add edge if it has not yet been added yet
        if G.has_edge(n1, n2):
            continue

        # Also avoid edges that now connect to the same node
        if n1 == n2:
            continue

        G.add_edge(n1, n2, length=length, mode=mode)

    # Now we can remove all edges and nodes that predated the
    # coalescing operations
    for n in reference.index:
        # Note that this will also drop all edges
        G.remove_node(n)

    # Also make sure to update the new nodes with their summary
    # stats and locational data
    for i, node in new_node_coords.items():
        if G.has_node(i):
            # For all other nodes, preserve them by re-populating
            for key in node:
                G.nodes[i][key] = node[key]

    return G


def _path_has_consistent_mode_type(G, path):
    # Makes sure that no mixed transit+walk network components
    # made it through the get_paths... method - we do not want to
    # mix modes during the simplification process
    path_modes = []
    for u, v in zip(path[:-1], path[1:]):
        edge_count = G.number_of_edges(u, v)
        for i in range(edge_count):
            edge = G.edges[u, v, i]
            path_modes.append(edge['mode'])
    path_clear = all(x == path_modes[0] for x in path_modes)
    return path_clear


def is_endpoint(G: nx.Graph, node: int, strict=True):
    """
    Return True if the node is a "real" endpoint of an edge in the network, \
    otherwise False. OSM data includes lots of nodes that exist only as \
    points to help streets bend around curves. An end point is a node that \
    either: \
    1) is its own neighbor, ie, it self-loops. \
    2) or, has no incoming edges or no outgoing edges, ie, all its incident \
        edges point inward or all its incident edges point outward. \
    3) or, it does not have exactly two neighbors and degree of 2 or 4. \
    4) or, if strict mode is false, if its edges have different OSM IDs.

    Please note this method is taken directly from OSMnx, and can be found in \
    its original form, here: \
    https://github.com/gboeing/osmnx/blob/ \
    c5916aab5c9b94c951c8fb1964c841899c9467f8/osmnx/simplify.py#L22-L88

    Parameters
    ----------
    G : networkx multidigraph
        The NetworkX graph being evaluated
    node : int
        The node to examine
    strict : bool
        If False, allow nodes to be end points even if they fail all other \
        rules  but have edges with different OSM IDs

    Returns
    -------
    bool
        Indicates whether or not the node is indeed an endpoint
    """

    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    if node in neighbors:
        # If the node appears in its list of neighbors, it self-loops. this is
        # always an endpoint.
        return True

    # If node has no incoming edges or no outgoing edges, it must be an
    # endpoint
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        return True

    elif not (n == 2 and (d == 2 or d == 4)):
        # Else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    elif not strict:
        # Non-strict mode
        osmids = []

        # Add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]['osmid'])

        # Add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]['osmid'])

        # If there is more than 1 OSM ID in the list of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    else:
        # If none of the preceding rules returned true, then it is not an
        # endpoint
        return False


def build_path(
        G: nx.Graph,
        node: int,
        endpoints: List[int],
        path: List[int]) -> List[int]:
    """
    Recursively build a path of nodes until you hit an endpoint node.

    Please note this method is taken directly from OSMnx, and can be found in \
    its original form, here: \
    https://github.com/gboeing/osmnx/blob/ \
    c5916aab5c9b94c951c8fb1964c841899c9467f8/osmnx/simplify.py#L91-L131

    Parameters
    ----------
    G : networkx multidigraph
    node : int
        the current node to start from
    endpoints : set
        the set of all nodes in the graph that are endpoints
    path : list
        the list of nodes in order in the path so far

    Returns
    -------
    paths_to_simplify : list
    """

    # For each successor in the passed-in node
    for successor in G.successors(node):
        if successor not in path:
            # If successor is already in path, ignore it, otherwise add to path
            path.append(successor)

            if successor not in endpoints:
                # If successor not endpoint, recursively call
                # build_path until endpoint found
                path = build_path(G, successor, endpoints, path)

            else:
                # If successor is endpoint, path is completed, so return
                return path

    if (path[-1] not in endpoints) and (path[0] in G.successors(path[-1])):
        # If end of the path is not actually an endpoint and the path's
        # first node is a successor of the path's final node, then this is
        # actually a self loop, so add path's first node to end of path to
        # close it
        path.append(path[0])

    return path


def get_paths_to_simplify(G: nx.Graph, strict: bool=True) -> List[List[int]]:
    """
    Create a list of all the paths to be simplified between endpoint nodes. \
    The path is ordered from the first endpoint, through the interstitial \
    nodes, to the second endpoint.

    Please note this method is taken directly from OSMnx, and can be found in \
    its original form, here: \
    https://github.com/gboeing/osmnx/blob/ \
    c5916aab5c9b94c951c8fb1964c841899c9467f8/osmnx/simplify.py#L134-L181

    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other \
        rules but have edges with different OSM IDs

    Returns
    -------
    paths_to_simplify : lists
        Returns a nested set of lists, containing the paths (node ID arrays) \
        for each group of vertices that can be consolidated
    """

    # First identify all the nodes that are endpoints
    endpoints = set([node for node in G.nodes()
                     if is_endpoint(G, node, strict=strict)])

    # Initialize the list to be returned; an empty list
    paths_to_simplify = []

    # For each endpoint node, look at each of its successor nodes
    for node in endpoints:
        for successor in G.successors(node):
            if successor not in endpoints:
                # if the successor is not an endpoint, build a path from the
                # endpoint node to the next endpoint node
                try:
                    paths_to_simplify.append(
                        build_path(G,
                                   successor,
                                   endpoints,
                                   path=[node, successor]))
                except RuntimeError:
                    # Note: Recursion errors occur if some connected component
                    #       is a self-contained ring in which all nodes are not
                    #       end points handle it by just ignoring that
                    #       component and letting its topology remain intact
                    #       (this should be a rare occurrence).
                    log(('Recursion error: exceeded max depth, moving on to '
                         'next endpoint successor'), level=lg.WARNING)

    return paths_to_simplify


def simplify_graph(G_orig: nx.MultiDiGraph) -> nx.MultiDiGraph:
    # Note: This operation borrows heavily from the operation of
    #       the same name in OSMnx, as it existed in this state/commit:
    #       github.com/gboeing/osmnx/blob/
    #       c5916aab5c9b94c951c8fb1964c841899c9467f8/osmnx/simplify.py
    #       Function on line 203

    # Prevent upstream mutation, always copy
    G = G_orig.copy()

    # Used to track updates to execute
    all_nodes_to_remove = []
    all_edges_to_add = []

    # TODO: Improve this method to not produce any mixed mode path
    #       removal proposals
    # Utilize the recursive function from OSMnx that identifies paths based
    # on isolated successor nodes
    paths_to_consider = get_paths_to_simplify(G)

    # Iterate through the resulting path arrays to target
    for path in paths_to_consider:
        # If the path is not all one mode of travel, skip the
        # proposed simplification
        if not _path_has_consistent_mode_type(G, path):
            continue

        # Keep track of the edges to be removed so we can
        # assemble a LineString geometry with all of them
        edge_attributes = {}

        # Work from the last edge through, "wrapped around," to the beginning
        for u, v in zip(path[:-1], path[1:]):
            # Should not be multiple edges between interstitial nodes
            only_one_edge = G.number_of_edges(u, v) == 1
            if not only_one_edge:
                log(('Multiple edges between "{}" and "{}" '
                     'found when simplifying').format(u, v))

            # We ask for the 0th edge as we assume there is only one
            edge = G.edges[u, v, 0]
            for key in edge:
                if key in edge_attributes:
                    # If key already exists in dict, append
                    edge_attributes[key].append(edge[key])
                else:
                    # Otherwise, initialize a list
                    edge_attributes[key] = [edge[key]]

        # Note: In peartree, we opt to not preserve any other elements;
        #       we only keep length, mode and - in the case of simplified
        #       geometries - the shape of the simplified route
        edge_attributes['mode'] = edge_attributes['mode'][0]
        edge_attributes['length'] = sum(edge_attributes['length'])

        # Construct the geometry from the points array
        points_array = []
        for node in path:
            p = Point((G.nodes[node]['x'], G.nodes[node]['y']))
            points_array.append(p)
        edge_attributes['geometry'] = LineString(points_array)

        # Add nodes and edges to respective lists for processing
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append({'origin': path[0],
                                 'destination': path[-1],
                                 'attr_dict': edge_attributes})

    # For each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge['origin'], edge['destination'], **edge['attr_dict'])

    # Remove all the interstitial nodes between the new edges, which will also
    # knock out the related edges from the graph
    G.remove_nodes_from(set(all_nodes_to_remove))

    # TODO: This step could be significantly optimized (as well as
    # parameterized, made optional)
    # A final step that cleans out all duplicate edges (not desired in a
    # simplified network)
    mult_edges = []
    mult_edges_full = []
    for fr, to, edge in G.edges(data=True):
        if G.number_of_edges(fr, to) > 1:
            mult_edges.append((fr, to))
            mult_edges_full.append((fr, to, edge))

    # Clean out the permutations to just one of each
    mult_edges = set(mult_edges)

    # TODO: This nested for loop is sloppy; clean up (numpy scalars, perhaps)
    for fr1, to1 in mult_edges:
        subset_edges = []
        for fr2, to2, edge in mult_edges_full:
            if fr1 == fr2 and to1 == to2:
                subset_edges.append(edge)
        keep = max(subset_edges, key=lambda x: x['length'])

        # Drop all the edges
        edge_ct = len(subset_edges)
        G.remove_edges_from([(fr1, to1)] * edge_ct)

        # Then just re-add the one that we want
        G.add_edge(fr1, to1, **keep)

    return G
