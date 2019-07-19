import abc
import warnings
from functools import partial
from typing import Any, Dict, Iterable, List, Optional

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import LineString, MultiPoint, Point, shape
from shapely.ops import linemerge, split, transform

from .toolkit import generate_random_name


def _generate_point_array_override(
        mp_array: Iterable[Point],
        route_shape: LineString,
        existing_graph_nodes: gpd.GeoDataFrame,
        stop_distance_distribution: float) -> Iterable[Point]:
    # TODO: Better parameterize the factor used with the distance distribution
    # Figure that we give a 10% "give" - this is, if a stop is within
    # 10% of the target segment distance, then it should be re-assigned
    # to that stop
    reasonable_distance = 0.5 * stop_distance_distribution

    # TODO: Better parameterize the buffer distance setting here
    # Find the nearby stops for the target line from those available
    buffer_distance = 10  # meters
    buffered = route_shape.buffer(buffer_distance)

    try:
        # Use spatial index if available
        egn_sindex = existing_graph_nodes.sindex
        possibles = list(egn_sindex.intersection(route_shape.bounds))
        existing_graph_nodes_sub = existing_graph_nodes.iloc[possibles]
        intersects_mask = existing_graph_nodes_sub.intersects(buffered)
        intersecting_stops_gdf = existing_graph_nodes_sub[intersects_mask]

    except Exception:
        # Otherwise skip the spatial index and just intersect
        # against all geometries in the GeoDataFrame
        intersects_mask = existing_graph_nodes.intersects(buffered)
        intersecting_stops_gdf = existing_graph_nodes[intersects_mask]

    # If we encounter a situation where there are no nearby nodes, bail
    # early and just retyurn the original input array
    if len(intersecting_stops_gdf) == 0:
        return mp_array

    # TODO: Right now we only consider the stops as points (shapes),
    #       but in the future should aim to reuse graph nodes that are
    #       the same location by passing along the node id
    intersecting_stops = intersecting_stops_gdf['geometry'].values

    mp_array_override = []
    for pt in mp_array:
        # TODO: This is can be super slow, is there a potentially
        #       faster path or is the fact that this is only a subset
        #       of all stop nodes mean that the number of points being
        #       evaluated is acceptably small?
        dists = intersecting_stops_gdf.distance(pt).values

        # Skip if the closest existing stop is still too far away
        if dists.min() > reasonable_distance:
            mp_array_override.append(pt)
            continue

        # TODO: This mask operation is slow - is there a faster way?
        dists_mask = dists <= dists.min()
        sub_g_nodes = intersecting_stops[dists_mask]

        # Handle edge cases where matches are not captured
        if len(sub_g_nodes) == 0:
            mp_array_override.append(pt)
            continue

        # Otherwise, we should pass the first nearest stop node
        # as the new stop location
        first_nearest = sub_g_nodes[0]

        # But still make sure that it is on the line because
        # we need to use the points to break up the line and get the
        # segment distances in the next step
        adj_projected = route_shape.project(first_nearest)
        adjusted_nearest = route_shape.interpolate(adj_projected,
                                                   normalized=True)
        mp_array_override.append(adjusted_nearest)

    return mp_array_override


def generate_meter_projected_chunks(
        route_shape: LineString,
        custom_stops: Optional[List[List[float]]]=None,
        stop_distance_distribution: int=None,
        from_proj='epsg:4326',
        to_proj='epsg:2163',
        existing_graph_nodes: Optional[pd.DataFrame]=None) -> List[LineString]:

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
        # Sanity check (this should never occur due to checks in class init)
        if stop_distance_distribution is None:
            raise ValueError(('Auto stop assignment triggered, but '
                              'stop_distance_distribution is Nonetype'))

        # Divide the target route into roughly equal length segments
        # and get the number that would be needed to accomplish this
        stop_count = round(rs2.length / stop_distance_distribution)

        # Create the array of break points/joints
        mp_array = []
        for i in range(1, stop_count):
            fr = (i / stop_count)
            mp_array.append(rs2.interpolate(fr, normalized=True))

        # At this point, we have an array of what might be described as
        # "potential stops." From these stops, we want to look to see if there
        # are nearby stop alternatives that are existing stops used by the
        # current network graph. If there are, then we should use those
        # instead.
        if existing_graph_nodes is not None and len(existing_graph_nodes) > 0:
            mp_array_override = _generate_point_array_override(
                mp_array,
                rs2,
                existing_graph_nodes,
                stop_distance_distribution)

            # Now that mp_array_override has been fully populated,
            # can now override the original array holding the estimated
            # stop point locations
            mp_array = mp_array_override

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
        headway: float,
        wait_time_cost_method: Any,
        mode: Optional[str]) -> pd.DataFrame:
    avg_costs = []
    stop_lats = []
    stop_lons = []
    mode_col = []

    default_avg_cost = headway / 2

    for point in all_points:
        avg_costs.append(default_avg_cost)
        stop_lats.append(point.y)
        stop_lons.append(point.x)

        # TODO: Improve addition of the mode list to the
        #       nodes dataframe
        if mode:
            mode_col.append([mode])
        else:
            # When no mode is provided just pass through empty list
            mode_col.append([])

    nodes_df = pd.DataFrame({
        'stop_id': stop_ids,
        'avg_cost': avg_costs,
        'stop_lat': stop_lats,
        'stop_lon': stop_lons,
        'modes': mode_col,
    })

    return nodes_df


def generate_edges_df(
        stop_ids: List[str],
        chunks: List[LineString],
        avg_speed: float) -> pd.DataFrame:
    from_stop_ids = []
    to_stop_ids = []
    edge_costs = []
    mode_col = []

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
        l_km = chunks[i].length / 1000  # distance in km

        # Note: Average speed is to be supplied in kmph
        in_hours = round(l_km / avg_speed, 3)
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

    if 'mode' in props:
        # GTFS mode types are held as strings
        fresh_props['mode'] = str(props['mode'])

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

    if 'mode' not in fresh_props:
        # Mode id based on GTFS mode spec - routes supplied absent modes
        # may lead to undesired effects
        warnings.warn(
            'No mode id supplied for synthetic route, leaving blank.')

    return fresh_props


class SyntheticTransitLine(abc.ABC):
    """
    Represents a single synthetic transit lines custom attributes and shape.

    Derived from a single Feature in a TransitJSON GeoJSON FeatureCollection.
    """

    def __init__(
            self,
            feature: Dict[str, Any],
            wait_time_cost_method: Any,
            existing_graph_nodes: Optional[pd.DataFrame]=None):
        # All values have defaults built in; which are overridden when the
        # user supplies, through the TransitJSON, custom values for those
        # properties.
        feature_props = feature['properties']
        props = _validate_feature_properties(feature_props)

        self._mode = props.get('mode', None)

        # Headway measured in seconds (30 minutes to seconds)
        self._headway = props.get('headway', 30 * 60)
        self._wait_time_cost_method = wait_time_cost_method

        # Speed is measured in miles per hour
        self._average_speed = props.get('average_speed', 8)

        self._bidirectional = props.get('bidirectional', False)

        # Via the validation step; one of these two will be set
        custom_stops = props.get('custom_stops', None)
        self.used_custom_stops = custom_stops is not None

        # Note that stops distances are set in meters (e.g. 402 meters
        # is the equivalent of every 1/4 of a mile)
        stop_distance = props.get('stop_dist', 402)

        # We require this GeoJSON coordinate component to be valid format
        self._route_path = shape(feature['geometry'])

        # Generate reference geometry data, note (and this is confusing) but
        # chunks is in meter projection and all_pts is in web mercator
        # this is because we only need (from chunks) the length value
        # and do not actually preserve the geometry beyond these operations
        chunks = generate_meter_projected_chunks(
            route_shape=self._route_path,
            custom_stops=custom_stops,
            stop_distance_distribution=stop_distance,
            existing_graph_nodes=existing_graph_nodes)

        # Generate stops from each chunk and assign each a unique id
        all_pts = generate_stop_points(chunks)
        stop_ids = generate_stop_ids(len(all_pts))

        # Produce key graph components
        self._nodes = generate_nodes_df(
            stop_ids,
            all_pts,
            self._headway,
            self._wait_time_cost_method,
            self._mode)
        self._edges = generate_edges_df(
            stop_ids,
            chunks,
            self._average_speed)

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

    def __init__(
            self,
            feature_collection: Dict[str, Any],
            wait_time_cost_method: Any,
            existing_graph_nodes: Optional[pd.DataFrame]=None):
        # Initialize an empty list
        self._lines = []

        # For each Feature in the FeatureCollection group; add an additional
        # instantiated SyntheticTransitLine object
        for feature in feature_collection['features']:
            new_line = SyntheticTransitLine(
                feature,
                wait_time_cost_method,
                existing_graph_nodes)
            self._lines.append(new_line)

    def _create_all_lines_generator(self):
        for line in self._lines:
            yield line

    def get_all_lines(self) -> Iterable[SyntheticTransitLine]:
        return self._create_all_lines_generator()

    lines = property(get_all_lines)
