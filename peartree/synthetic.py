from functools import partial
from typing import Dict, List

import pyproj
from shapely.geometry import LineString, MultiPoint, Point, shape
from shapely.ops import linemerge, split, transform

from .paths import _generate_random_name


def generate_meter_projected_chunks(
        route_shape: LineString,
        stop_distance_distribution: int) -> List[LineString]:
    
    # Reproject 4326 lat/lon coordinates to equal area
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), # source coordinate system
        pyproj.Proj(init='epsg:2163')) # destination coordinate system

    rs2 = transform(project, route_shape)  # apply projection
    stop_count = round(rs2.length / stop_distance_distribution)

    # Create the array of break points/joints
    mp_array = []
    for i in range(1, stop_count):
        fr = (i/stop_count)
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


def generate_stop_points(
        route_shape: LineString,
        stop_distance_distribution: int):
    # Rename variable for brevity
    rs = route_shape
    
    # Create the array of break points/joints
    stop_count = round(rs.length / stop_distance_distribution)
    mp_array = []
    for i in range(1, stop_count):
        fr = (i/stop_count)
        mp_array.append(rs.interpolate(fr, normalized=True))
    
    # Resulting array is compose of first and last point, plus
    # splitter points in the middle
    all_points = [Point(rs.coords[0])]
    all_points += mp_array
    all_points += [Point(rs.coords[-1])]
    return all_points


def generate_stop_ids(stops_count: int) -> List[str]:
    shape_name = _generate_random_name(5)
    stop_names = []
    for i in range(stops_count):
        stop_names.append('_'.join([shape_name, i]))
    return stop_names


def generate_nodes_df(
        stop_ids: List[str],
        all_points: List[Point],
        headway: float) -> pd.DataFrame:
    avg_costs = []
    stop_lats = []
    stop_lons = []

    default_avg_cost = headway/2

    for point in all_points:
        avg_costs.append(default_avg_cost)
        stop_lats.append(point.x)
        stop_lons.append(point.y)

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
                        'correct route shape subdivisions.')

    for i, nodes in enumerate(paired_nodes):
        point_a = nodes[0]
        point_b = nodes[1]
        from_stop_ids.append(point_a)
        to_stop_ids.append(point_b)
        
        # Estimate the amount of time it would
        # take to traverse this portion of the
        # route path given the default speed
        l = clean_chunks[i].length / 1000  # distance in km
        # Note: Average speed is to be supplied in kmph
        edge_costs.append(l / avg_speed)

    edges_df = pd.DataFrame({
        'from_stop_id': from_stop_ids,
        'to_stop_id': to_stop_ids,
        'edge_cost': edge_costs,
    })
    
    return edges_df
