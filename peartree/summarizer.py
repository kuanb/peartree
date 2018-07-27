import multiprocessing as mp
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import partridge as ptg
from peartree.parallel import (RouteProcessor, TripTimesInterpolator,
                               make_route_processor_manager,
                               make_trip_time_interpolator_manager)
from peartree.utilities import log



class InvalidParsedWaitTimes(Exception):
    pass

def _format_summarized_outputs(summarized: pd.Series) -> pd.DataFrame:
    # The output of the group by produces a Series, but we want to extract
    # the values from the index and the Series itself and generate a
    # pandas DataFrame instead
    original_stop_ids_index = summarized.index.values
    original_series_values = summarized.values

    return pd.DataFrame({
        'stop_id': original_stop_ids_index,
        'avg_cost': original_series_values})


def calculate_average_wait(direction_times: pd.DataFrame) -> float:
    # Exit early if we do not have enough values to calculate a mean
    at = direction_times.arrival_time
    if len(at) < 2:
        return np.nan

    first = at[1:].values
    second = at[:-1].values
    wait_seconds = (first - second)

    # TODO: Can implement something more substantial here that takes into
    #       account divergent/erratic performance or intentional timing
    #       clusters that are not evenly dispersed
    na = np.array(wait_seconds)
    average_wait = na.mean()
    return average_wait


def summarize_edge_costs(df: pd.DataFrame) -> pd.DataFrame:
    # Used as a function applied to a grouping
    # operation, pulls out the mean edge cost for each
    # unqiue edge pair (from node and to node)
    from_stop_id = df.from_stop_id.values[0]
    results_mtx = []
    for to_stop_id in df.to_stop_id.unique():
        to_mask = (df.to_stop_id == to_stop_id)
        avg_cost = df[to_mask].edge_cost.mean()
        results_mtx.append([avg_cost,
                            from_stop_id,
                            to_stop_id])
    return pd.DataFrame(results_mtx, columns=df.columns)


def generate_summary_edge_costs(all_edge_costs: pd.DataFrame) -> pd.DataFrame:
    # Given a dataframe of edges costs, get the average for each
    # from node - to node pair
    summary_groupings = all_edge_costs.groupby('from_stop_id')
    summary = summary_groupings.apply(summarize_edge_costs)
    summary = summary.reset_index(drop=True)
    return summary


def summarize_waits_at_one_stop(stop_df: pd.DataFrame) -> float:
    # Calculates average wait time at this stop, given all observed
    # TODO: Simply dividiing by two may not be appropriate - it is
    #       go for estimation purposes, but I could introduce
    #       more sophisticated wait time calculations here
    divide_by = (len(stop_df) * 2)
    dir_0_sum = stop_df.wait_dir_0.sum()
    dir_1_sum = stop_df.wait_dir_1.sum()

    # A weighted average is performed, which could inaccurately8
    # portrary a wait time at a given stop if one direction has
    # significantly higher frequence than another
    calculated = ((dir_0_sum + dir_1_sum) / divide_by)

    return calculated


def generate_summary_wait_times(
        df: pd.DataFrame,
        fallback_stop_cost: float) -> pd.DataFrame:
    df_sub = df[['stop_id',
                 'wait_dir_0',
                 'wait_dir_1']].reset_index(drop=True)
    init_of_stop_ids = df_sub.stop_id.unique()

    # Default values for average waits with not enough data should be
    # NaN types, but let's make sure all null types are NaNs to be safe
    for col in ['wait_dir_0', 'wait_dir_1']:
        mask = df_sub[col].isnull()
        df_sub.loc[mask, col] = np.nan

        # Convert anything that is 0 or less seconds to a NaN as well
        # to remove negative or 0 second waits in the system
        df_sub.loc[~(df_sub[col] > 0), col] = np.nan

        # With all null types converted to NaN, we can cast col as float
        df_sub[col] = df_sub[col].astype(float)

    # Clean out the None values
    dir_0_mask = ~np.isnan(df_sub.wait_dir_0)
    dir_1_mask = ~np.isnan(df_sub.wait_dir_1)

    # We can't include values where both directions
    # have NaNs at same time
    d0_ids = df_sub[dir_0_mask].stop_id.unique()
    d1_ids = df_sub[dir_1_mask].stop_id.unique()
    keep_ids = list(d0_ids) + list(d1_ids)
    df_sub_clean = df_sub[df_sub.stop_id.isin(keep_ids)]

    orig_len = len(df_sub)
    new_len = len(df_sub_clean)
    if not new_len == orig_len:
        log(('Cleaned out bi-directional NaN values from '
             'stop IDs. From {} to {}.'.format(orig_len, new_len)))
        # And now replace df_sub
        df_sub = df_sub_clean

    # Recheck all for NaNs
    dir_0_mask_2 = np.isnan(df_sub.wait_dir_0)
    dir_1_mask_2 = np.isnan(df_sub.wait_dir_1)

    df_sub.loc[dir_0_mask_2, 'wait_dir_0'] = df_sub.wait_dir_1
    df_sub.loc[dir_1_mask_2, 'wait_dir_1'] = df_sub.wait_dir_0

    # TODO: All this pruning is a mess, needs to be
    #       organized much better
    # One more time to drop out the subset that are NaN
    # from a given stop id
    dir_0_mask_3 = ~np.isnan(df_sub.wait_dir_0)
    df_sub = df_sub[dir_0_mask_3]

    dir_1_mask_3 = ~np.isnan(df_sub.wait_dir_1)
    df_sub = df_sub[dir_1_mask_3]

    # Make sure that there are no None values left
    dir_0_check_2 = df_sub[np.isnan(df_sub.wait_dir_0)]
    dir_1_check_2 = df_sub[np.isnan(df_sub.wait_dir_1)]

    dir_0_trigger = len(dir_0_check_2) > 0
    dir_1_trigger = len(dir_1_check_2) > 0
    if dir_0_trigger or dir_1_trigger:
        raise InvalidParsedWaitTimes(
            'NaN values for both directions on some stop IDs.')

    # At this point, we should make sure that there are still values
    # in the DataFrame - otherwise we are in a situation where there are
    # no valid times to evaluate. This is okay; we just need to skip straight
    # to the application of the fallback value
    if df_sub.empty:
        # So just make a fallback empty dataframe for now
        summed_reset = pd.DataFrame({'stop_id': [], 'avg_cost': []})

    # Only attempt this group by summary if at least one row to group on
    else:
        grouped = df_sub.groupby('stop_id')
        summarized = grouped.apply(summarize_waits_at_one_stop)

        # Clean up summary results, reformat pandas DataFrame result
        summed_reset = _format_summarized_outputs(summarized)

    end_of_stop_ids = summed_reset.stop_id.unique()
    log('Original stop id count: {}'.format(len(init_of_stop_ids)))
    log('After cleaning stop id count: {}'.format(len(end_of_stop_ids)))

    # Check for the presence of any unresolved stop ids and
    # assign them some value boarding cost
    if len(init_of_stop_ids) > len(end_of_stop_ids):
        a = set(list(init_of_stop_ids))
        b = set(list(end_of_stop_ids))
        unresolved_ids = list(a - b)
        log('Some unaccounted for stop ids. '
            'Resolving {}...'.format(len(unresolved_ids)))

        # TODO: Perhaps these are start/end stops and should adopt
        #       a cost that is "average" for that route?
        #       I should think of how to actually do this
        #       because we do not have enough data, for now let's
        #       just assign some default high cost connection value
        #       to these stops
        sids = list(summed_reset.stop_id)
        acst = list(summed_reset.avg_cost)
        for i in unresolved_ids:
            sids.append(i)
            acst.append(fallback_stop_cost)

        # Rebuild the dataframe
        summed_reset = pd.DataFrame({'stop_id': sids, 'avg_cost': acst})

    return summed_reset


def _trip_times_interpolator_pool_map(
        trip_times_interpolator_proxy: RouteProcessor,
        target_trip_id: str):
    return trip_times_interpolator_proxy.generate_infilled_times(
        target_trip_id)


def _linearly_interpolate_infill_times(
        stop_times_orig_df: pd.DataFrame,
        use_multiprocessing: bool) -> pd.DataFrame:
    # Prevent any upstream modification of this object
    stops_times_df = stop_times_orig_df.copy()

    # Extract a list of all unqiue trip ids attached to the stops
    target_trip_ids = stops_times_df['trip_id'].unique().tolist()

    # Monitor run time performance
    start_time = time.time()
    if use_multiprocessing is True:
        cpu_count = mp.cpu_count()
        log('Running parallelized trip times interpolation on '
            '{} processes'.format(cpu_count))

        manager = make_trip_time_interpolator_manager()
        trip_times_interpolator = manager.TripTimesInterpolator(stops_times_df)

        with mp.Pool(processes=cpu_count) as pool:
            results = pool.starmap(_trip_times_interpolator_pool_map,
                                   [(trip_times_interpolator, trip_id)
                                    for trip_id in target_trip_ids])
    else:
        log('Running serialized trip times interpolation (no parallelization)')
        trip_times_interpolator = TripTimesInterpolator(stops_times_df)
        results = [trip_times_interpolator.generate_infilled_times(trip_id)
                   for trip_id in target_trip_ids]
    elapsed = round(time.time() - start_time, 2)
    log('Trip times interpolation complete. Execution time: {}s'.format(
        elapsed))

    # Take all the resulting dataframes and stack them together
    cleaned = []
    for times_sub in results:
        # Note: Extract values as list with the intent of avoiding
        #       otherwise-expensive append operations (df-to-df)
        cleaned.extend(times_sub.values.tolist())

    # Prepare for new df creation by getting list of columns
    cols = stops_times_df.columns.values.tolist()
    cols.remove('trip_id')
    cols.append('trip_id')

    # Convert matrices to a pandas DataFrame again
    cleaned_new_df = pd.DataFrame(cleaned, columns=cols)
    cleaned_new_df = cleaned_new_df.reset_index(drop=True)

    return cleaned_new_df


def _route_analyzer_pool_map(
        route_analyzer_proxy: RouteProcessor,
        target_route_id: str):
    return route_analyzer_proxy.generate_route_costs(target_route_id)


def _generate_route_processing_results(
        target_route_ids: List,
        target_time_start: int,
        target_time_end: int,
        ftrips: pd.DataFrame,
        stop_times: pd.DataFrame,
        feed_stops: pd.DataFrame,
        use_multiprocessing: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Track the runtime of this method
    start_time = time.time()

    if use_multiprocessing is True:
        cpu_count = mp.cpu_count()
        log('Running parallelized route costing on '
            '{} processes'.format(cpu_count))

        manager = make_route_processor_manager()
        route_analyzer = manager.RouteProcessor(
            target_time_start,
            target_time_end,
            ftrips,
            stop_times,
            feed_stops)

        with mp.Pool(processes=cpu_count) as pool:
            results = pool.starmap(_route_analyzer_pool_map,
                                   [(route_analyzer, route_id)
                                    for route_id in target_route_ids])
    else:
        log('Running serialized route costing (no parallelization)')
        route_analyzer = RouteProcessor(
            target_time_start,
            target_time_end,
            ftrips,
            stop_times,
            feed_stops)
        results = [route_analyzer.generate_route_costs(rid)
                   for rid in target_route_ids]
    elapsed = round(time.time() - start_time, 2)
    log('Route costing complete. Execution time: {}s'.format(elapsed))

    # First, create a 2-dimensional matrix for each of the output series
    all_edge_costs = []
    all_wait_times = []

    for tst_sub, edge_costs in results:

        # For each result, skip if it is empty
        if type(edge_costs) is pd.DataFrame and not edge_costs.empty:
            # Resume the expected adding of each list result to the matrices
            all_edge_costs.extend(edge_costs.values.tolist())

        # And again, for the other dataframe
        if type(tst_sub) is pd.DataFrame and not tst_sub.empty:
            all_wait_times.extend(tst_sub.values.tolist())

    # Convert matrices to a pandas DataFrame again
    all_edge_costs_columns = ['edge_cost', 'from_stop_id', 'to_stop_id']
    all_edge_costs_new_df = pd.DataFrame(all_edge_costs,
                                         columns=all_edge_costs_columns)

    all_wait_times_columns = ['stop_id', 'wait_dir_0', 'wait_dir_1']
    all_wait_times_new_df = pd.DataFrame(all_wait_times,
                                         columns=all_wait_times_columns)

    return (all_edge_costs_new_df, all_wait_times_new_df)


def _trim_stop_times_by_timeframe(
        init_stop_times_orig: pd.DataFrame,
        target_time_start: int,
        target_time_end: int) -> pd.DataFrame:
    # Trim down stop_times df based on requested time range
    # before feeding into the interpolation step downstream
    init_stop_times = init_stop_times_orig.copy()

    # Create masks for time range
    start_time_mask = (init_stop_times.arrival_time >= target_time_start)
    end_time_mask = (init_stop_times.arrival_time <= target_time_end)

    # Select stop times within the range (satisfies both mask constraints)
    both_mask = (start_time_mask & end_time_mask)
    within_timeframe_sub = init_stop_times[both_mask]

    # Get unique trip ids associated with those stops
    want_trip_ids = within_timeframe_sub.trip_id.unique()

    # If any of the stop of a given trip id is the requested time range,
    # perserve all the stops in that trip
    want_trips_mask = init_stop_times.trip_id.isin(want_trip_ids)
    sub_stop_times_final = init_stop_times[want_trips_mask]

    return sub_stop_times_final


def generate_edge_and_wait_values(
        feed: ptg.gtfs.feed,
        target_time_start: int,
        target_time_end: int,
        interpolate_times: bool,
        use_multiprocessing: bool) -> Tuple[pd.DataFrame]:
    sub_stop_times = _trim_stop_times_by_timeframe(
        feed.stop_times, target_time_start, target_time_end)

    # Flags whether we interpolate intermediary stops or not
    if interpolate_times:
        # Prepare the stops times dataframe by also infilling
        # all stop times that are NaN with their linearly interpolated
        # values based on their nearest numerically valid neighbors
        stop_times = _linearly_interpolate_infill_times(
            sub_stop_times,
            use_multiprocessing)
    else:
        stop_times = sub_stop_times.copy()

    # Initialize the trips dataframe to be worked with
    ftrips = feed.trips.copy()
    ftrips = ftrips[~ftrips['route_id'].isnull()]

    # Execute the route-level processing operations with prepped data
    (all_edge_costs,
     all_wait_times) = _generate_route_processing_results(
        feed.routes.route_id,
        target_time_start,
        target_time_end,
        ftrips,
        stop_times,
        feed.stops.copy(),
        use_multiprocessing)

    return (all_edge_costs, all_wait_times)
