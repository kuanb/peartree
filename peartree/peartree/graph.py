import networkx as nx
import numpy as np
import pandas as pd
import random

from utilities import log


def generate_empty_md_graph(name,
                            init_crs={'init':'epsg:4326'}):
    return nx.MultiDiGraph(name=name, crs=init_crs)


def nameify_stop_id(name, sid):
    name = str(name)
    sid = str(sid)
    return f'{name}_{sid}'


def populate_graph(G,
                   name,
                   feed,
                   wait_times_by_stop,
                   summary_edge_costs):
    for i, row in wait_times_by_stop.iterrows():
        sid = str(row.stop_id)
        full_sid = f'{name}_{sid}'

        # TODO: Join tables before hand to make
        #       this part go faster
        id_mask = (feed.stops.stop_id==sid)
        stop_data = feed.stops[id_mask].head(1).T.squeeze()

        G.add_node(full_sid,
                   boarding_cost=row.avg_cost,
                   y=stop_data.stop_lat,
                   x=stop_data.stop_lon)

    for i, row in summary_edge_costs.iterrows():
        sid_fr = nameify_stop_id(name, row.from_stop_id)
        sid_to = nameify_stop_id(name, row.to_stop_id)
        G.add_edge(sid_fr,
                   sid_to,
                   length=row.edge_cost)

    return G