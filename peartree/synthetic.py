import abc
from functools import partial
from typing import Any, Dict, Iterable, List

import pandas as pd
import pyproj
from shapely.geometry import LineString, MultiPoint, Point, shape
from shapely.ops import linemerge, split, transform

from .toolkit import generate_random_name


def generate_meter_projected_chunks(
        route_shape: LineString,
        custom_stops: List[List[float]]=None,
        stop_distance_distribution: int=None,
        from_proj='epsg:4326',
        to_proj='epsg:2163') -> List[LineString]:

    # Reproject 4326 lat/lon coordinates to equal area
    project = partial(
        pyproj.transform,
        # source coordinate system
        pyproj.Proj(init=from_proj, preserve_units=True),
        # destination coordinate system
        pyproj.Proj(init=to_proj, preserve_units=True))

    rs2 = transform(project, route_shape)  # apply projection

    # Two ways to break apart this route into chunks:
    #   1. Using custom stops as break points (this one takes precedence)
    #   2. Using a custom distance to segment out the route

    # In either case, we need to generate mp_array such that we have
    # target stops or "break points" for the route line shape

    # Path 1 if available
    if custom_stops is not None:
        mp_array = []
        for custom_stop in custom_stops:
            # Now reproject with cast point geometry
            custom_stop_proj = transform(project, Point(custom_stop))
            interp_stop = rs2.interpolate(rs2.project(custom_stop_proj))
            mp_array.append(interp_stop)

    # Otherwise we go with path 2
    else:
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


def _validate_feature_properties(props: Dict) -> Dict:
    fresh_props = {}

    if 'headway' in props:
        fresh_props['headway'] = float(props['headway'])

    if 'average_speed' in props:
        fresh_props['average_speed'] = float(props['average_speed'])

    if 'bidirectional' in props:
        fresh_props['bidirectional'] = bool(props['bidirectional'])

    # For this section, either the custom stops of the stop distance
    # value must be set
    if 'stops' in props:
        # Make sure that this value is supplied as a list
        if isinstance(props['stops'], list):
            fresh_props['custom_stops'] = props['stops']

    if 'stop_distance_distribution' in props:
        fresh_props['stop_dist'] = float(props['stop_distance_distribution'])

    # Sanity check; if both custom stops and stops distance are None
    # then we cannot proceed
    no_stops = 'custom_stops' not in fresh_props
    no_dist = 'stop_dist' not in fresh_props
    if no_stops and no_dist:
        raise ValueError('Synthetic network addition must have either '
                         'custom stops or stops distance default set.')

    return fresh_props


class SyntheticTransitLine(abc.ABC):
    """
    Represents a single synthetic transit lines custom attributes and shape.

    Derived from a single Feature in a TransitJSON GeoJSON FeatureCollection.
    """

    def __init__(self, feature: Dict[str, Any]):
        # All values have defaults built in; which are overridden when the
        # user supplies, through the TransitJSON, custom values for those
        # properties.
        feature_props = feature['properties']
        props = _validate_feature_properties(feature_props)

        # Headway measured in seconds (30 minutes to seconds)
        self._headway = props.get('headway', 30 * 60)

        # Speed is measured in miles per hour
        self._average_speed = props.get('average_speed', 8)

        self._bidirectional = props.get('bidirectional', False)

        # Via the validation step; one of these two will be set
        custom_stops = props.get('custom_stops', None)
        # Note that stops distances are set in meters (e.g. 402 meters
        # is the equivalent of every 1/4 of a mile)
        stop_distance = props.get('stop_dist', 402)

        # We require this GeoJSON coordinate component to be valid format
        self._route_path = shape(feature['geometry'])

        # Generate reference geometry data, note (and this is confusing) but
        # chunks is in meter projection and all_pts is in web mercator
        # this is because we only need (from chunks) the length value
        # and do not actually preserve the geometry beyond these operations
        chunks = generate_meter_projected_chunks(self._route_path,
                                                 custom_stops,
                                                 stop_distance)

        # Generate stops from each chunk and assign each a unique id
        all_pts = generate_stop_points(chunks)
        stop_ids = generate_stop_ids(len(all_pts))

        # Produce key graph components
        self._nodes = generate_nodes_df(stop_ids, all_pts, self._headway)
        self._edges = generate_edges_df(stop_ids, chunks, self._average_speed)

    def get_nodes(self) -> pd.DataFrame:
        # Do this to prevent upstream mutation of the reference DataFrame
        return self._nodes.copy()

    def get_edges(self) -> pd.DataFrame:
        # Do this to prevent upstream mutation of the reference DataFrame
        return self._edges.copy()

    def get_bidrectional(self) -> bool:
        # Always attempt to avoid mutation
        return bool(self._bidirectional)

    nodes = property(get_nodes)
    edges = property(get_edges)
    bidirectional = property(get_bidrectional)


class SyntheticTransitNetwork(abc.ABC):
    """
    Holds a list of SyntheticTransitLine

    Below is an example of the most basic TransitJSON that could/should be \
    provided to peartree for modeling out a new transit line:

    {
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-82.7, 39.5], [-82.8, 39.6], [-82.9, 39.7] ]
                }, "properties": {
                        "average_speed": 16,
                        "headway": 600,
                        "stops": [ [-82.8, 39.6] ],
                }, "type": "Feature"
            }
        ],
    }
    """

    def __init__(self, feature_collection: Dict[str, Any]):
        # Initialize an empty list
        self._lines = []

        # For each Feature in the FeatureCollection group; add an additional
        # instantiated SyntheticTransitLine object
        for feature in feature_collection['features']:
            new_line = SyntheticTransitLine(feature)
            self._lines.append(new_line)

    def _create_all_lines_generator(self):
        for line in self._lines:
            yield line

    def get_all_lines(self) -> Iterable[SyntheticTransitLine]:
        return self._create_all_lines_generator()

    lines = property(get_all_lines)
