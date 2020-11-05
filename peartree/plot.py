import networkx as nx

from .utilities import log


def generate_plot(G: nx.MultiDiGraph, use_agg=False):
    # Load matplotlib only when plot requested
    import matplotlib  # noqa
    if use_agg:
        # Force matplotlib to not use any Xwindows backend
        matplotlib.use('Agg')

    # OSMnx is not a dependency anymore, so we should only allow the plot
    # function to work as a convenience, if the user has already installed
    # OSMnx
    try:
        import osmnx as ox  # noqa
    except ModuleNotFoundError:
        log(('Optional dependency: OSMnx must be installed to use the '
             'plot method in peartree'))

    # TODO: Build out custom plotting configurations but,
    #       in the meantime, use OSMnx's plotting configurations
    #       since they work well for the current use case and I
    #       also plan on incorporating OSMnx into this library
    #       down the road so it isn't too extraneous an import.
    fig, ax = ox.plot_graph(G,
                            figsize=(12,12),
                            show=False,
                            close=False,
                            node_color='#8aedfc',
                            node_size=5,
                            edge_color='#e2dede',
                            edge_alpha=0.25,
                            bgcolor='black')
    return (fig, ax)
