import os
import sys
from typing import Any, Tuple

import networkx as nx


def _import_graph_tool():
    # Naively assume everything is set up try and see if can just be imported
    try:
        import graph_tool as gt
        return gt
    except ModuleNotFoundError:
        pass

    # Note: graph-tool installs to a specific directory, not what
    #       is being used by Python as a default.

    # If we fail to import it the first time, check to see if it
    # has been downloaded in its default apt-get install directory
    sys.path.append(os.environ['GRAPH_TOOL_DIR'])

    # Now retry with the new system path appended
    try:
        import graph_tool as gt
        return gt
    except ModuleNotFoundError as e:
        # If still fails, pass the exception through a custom error
        raise GraphToolNotImported(
            'graph-tool was not able to be imported: {}'.format(e))

    return gt


class GraphToolNotImported(Exception):
    # Let's have a custom exception for when we read in GTFS files
    pass


def get_prop_type(value: Any, key: Any=None) -> Tuple[str, Any, str]:
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.

    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # Ensure that key is returned as a str type
    if isinstance(key, bytes):
        key = key.decode()

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, bytes):
        tname = 'string'
        value = value.decode()

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    # Final check to make sure outputs that need to be of a specific type are
    tname = str(tname)
    key = str(key)

    # Return all three results
    return tname, value, key


def nx_to_gt(nxG: nx.MultiDiGraph):
    """
    Converts a networkx graph to a graph-tool graph (gt.Graph).

    Credit: Please note that this function is adapted from Github user \
            @bbengfort's blog post 'Converting NetworkX to Graph-Tool', \
            available at the URL: \
                > https://bbengfort.github.io/snippets/\
                  2016/06/23/graph-tool-from-networkx.html \

    More information about this method available on @kuanbutts blog: \
        > http://kuanbutts.com/2018/08/17/peartree-to-graph-tool/

    Parameters
    ----------
        nxG : nx.MultiDiGraph
            The peartree network graph result of a processed set of GTFS feeds

    Returns
    ——
    gtG : graph_tool.Graph
        The converted network graph, instantiated as a graph_tool network graph
    """
    # First, attempt to import graph-tool
    gt = _import_graph_tool()

    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)  # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set()  # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):
        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            # Check: Skip properties already added
            if key in nprops:
                continue

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            # Create the PropertyMap, and...
            prop = gtG.new_vertex_property(tname)

            # ...then set the PropertyMap
            gtG.vertex_properties[key] = prop

            # Finally, add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set()  # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them,
        # just like was performed with the vertices attributes
        for key, val in data.items():
            # Check: Skip properties already added
            if key in eprops:
                continue

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            # Create the PropertyMap, and...
            prop = gtG.new_edge_property(tname)

            # ...then set the PropertyMap
            gtG.edge_properties[key] = prop

            # Finally, add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Vertex mapping (lookup) for tracking edges later
    vertices = {}
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex(n=1)
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            tname, value, key = get_prop_type(value, key)
            gtG.vp[key][v] = value  # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value  # ep is short for edge_properties

    # Done, finally!
    return gtG
