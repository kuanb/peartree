import matplotlib
import networkx as nx
import osmnx as ox

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


def generate_plot(G: nx.MultiDiGraph):
    # TODO: Build out custom plotting configurations but,
    #       in the meantime, use OSMnx's plotting configurations
    #       since they work well for the current use case and I
    #       also plan on incorporating OSMnx into this library
    #       down the road so it isn't too extraneous an import.
    fig, ax = ox.plot_graph(G,
                            fig_height=12,
                            show=False,
                            close=False,
                            node_color='#8aedfc',
                            node_size=5,
                            edge_color='#e2dede',
                            edge_alpha=0.25,
                            bgcolor='black')
    return (fig, ax)
