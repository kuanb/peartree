import collections

import networkx as nx


def convert_to_digraph(G_orig: nx.MultiDiGraph) -> nx.DiGraph:
    # Prevent upstream impacts
    G = G_orig.copy()

    dupes_dict = {}
    for node_id in G.nodes():
        nodes_to = []
        for fr, to in G.out_edges(node_id):
            nodes_to.append(to)
        to_collection = collections.Counter(nodes_to).items()
        dupes = [item for item, count in to_collection if count > 1]

        if len(dupes) > 0:
            dupes_dict[node_id] = {}

            for dupe in dupes:
                in_consideration = []

                # Get all the edge attributes for this node pair
                dupe_count = G.number_of_edges(node_id, dupe)
                for i in range(dupe_count):
                    e = G.edges[node_id, dupe, i]
                    in_consideration.append(e)

                # From the results, we optimistically select the fastest
                # edge value and all associated key/values from the list
                fastest_e = min(in_consideration, key=lambda x: x['length'])
                dupes_dict[node_id][dupe] = fastest_e

    # Now that we have a list of issue duplicates, we can
    # iterate through the list and remove and replace edges
    for fr in dupes_dict.keys():
        to_dict = dupes_dict[fr]
        for to in to_dict.keys():
            # Remove all the edges that exist, we are going
            # to start with a fresh slate (also, NetworkX makes
            # it really hard to control which edges you are
            # removing, otherwise)
            for i in range(G.number_of_edges(fr, to)):
                G.remove_edge(fr, to)

            # Now let's start fresh and add a new, single, edge
            G.add_edge(fr, to, **to_dict[to])

    # Now we should be safe to return a clean directed graph object
    return nx.DiGraph(G)
