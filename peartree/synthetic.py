from functools import partial
from typing import List

import pandas as pd
import pyproj
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import linemerge, split, transform

from .toolkit import generate_random_name


def generate_meter_projected_chunks(
        route_shape: LineString,
        stop_distance_distribution: int) -> List[LineString]:

    # Reproject 4326 lat/lon coordinates to equal area
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:2163'))  # destination coordinate system

    rs2 = transform(project, route_shape)  # apply projection
    stop_count = round(rs2.length / stop_distance_distribution)

    # Create the array of break points/joints
    mp_array = []
    for i in range(1, stop_count):
        fr = (i / stop_count)
        mp_array.append(rs2.interpolate(fr, normalized=True))

    # Cast array as a Shapely object
    splitter = MultiPoint(mp_array)

    # 1 meter buffer to address floating point discrepencies
    chunks = split(rs2, splitter.buffer(1))

    # TODO: Potential for length errors with this 1 meter
    #       threshold check

    # Take chunks and merge in the small lines
    # from intersection inside of the buffered circles
    # and attach to nearest larger line
    clean_chunks = [chunks[0]]
    r = len(chunks)
    for c in range(1, r):
        latest = clean_chunks[-1]
        current = chunks[c]
        # Again, this is a week point of the buffer of
        # 1 meter method
        if latest.length <= 2:
            # Merge in the small chunks with the larger chunks
            clean_chunks[-1] = linemerge([latest, current])
        else:
            clean_chunks.append(current)

    return clean_chunks


def generate_stop_points(chunks: List[LineString]) -> List[Point]:
    all_points = []

    # First point is the first node on the first line
    first_chunk = chunks[0].coords
    first_pt = [Point(p) for p in first_chunk][0]
    all_points.append(first_pt)

    # Then for all other points, we get from the end of each line
    for chunk in chunks:
        last_pt = [Point(p) for p in chunk.coords][-1]
        all_points.append(last_pt)

    # Now we need to convert to a Shapely object
    ap_ma = MultiPoint(all_points)

    # So we can reproject back out of equal area to 4326
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:2163'),  # source coordinate system
        pyproj.Proj(init='epsg:4326'))  # destination coordinate system

    ap_ma_reproj = transform(project, ap_ma)  # apply projection

    # Final step will be to pull out all points back into a list
    return [p for p in ap_ma_reproj]


def generate_stop_ids(stops_count: int) -> List[str]:
    shape_name = generate_random_name(5)
    stop_names = []
    for i in range(stops_count):
        stop_names.append('_'.join([shape_name, str(i)]))
    return stop_names


def generate_nodes_df(
        stop_ids: List[str],
        all_points: List[Point],
        headway: float) -> pd.DataFrame:
    avg_costs = []
    stop_lats = []
    stop_lons = []

    default_avg_cost = headway / 2

    for point in all_points:
        avg_costs.append(default_avg_cost)
        stop_lats.append(point.y)
        stop_lons.append(point.x)

    nodes_df = pd.DataFrame({
        'stop_id': stop_ids,
        'avg_cost': avg_costs,
        'stop_lat': stop_lats,
        'stop_lon': stop_lons,
    })

    return nodes_df


def generate_edges_df(
        stop_ids: List[str],
        all_points: List[Point],
        chunks: List[LineString],
        avg_speed: float) -> pd.DataFrame:
    from_stop_ids = []
    to_stop_ids = []
    edge_costs = []

    paired_nodes = list(zip(stop_ids[:-1], stop_ids[1:]))

    # Sanity check
    if not len(chunks) == len(paired_nodes):
        raise Exception('Chunking operation did not result '
                        'correct route shape subdivisions. '
                        '\nChunk len: {} \nPaired len: {}'.format(
                            str(len(chunks)),
                            str(len(paired_nodes))))

    for i, nodes in enumerate(paired_nodes):
        point_a = nodes[0]
        point_b = nodes[1]
        from_stop_ids.append(point_a)
        to_stop_ids.append(point_b)

        # Estimate the amount of time it would
        # take to traverse this portion of the
        # route path given the default speed
        l = chunks[i].length / 1000  # distance in km

        # Note: Average speed is to be supplied in kmph
        in_hours = round(l / avg_speed, 3)
        # Convert to seconds
        in_seconds = in_hours * 60 * 60
        edge_costs.append(in_seconds)

    edges_df = pd.DataFrame({
        'from_stop_id': from_stop_ids,
        'to_stop_id': to_stop_ids,
        'edge_cost': edge_costs,
    })

    return edges_df
