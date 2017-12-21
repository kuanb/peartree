from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import partridge as ptg

from .utilities import log


def calculate_average_wait(direction_times: pd.DataFrame) -> float:
    first = direction_times.arrival_time[1:].values
    second = direction_times.arrival_time[:-1].values
    wait_seconds = (first - second)

    # TODO: Can implement something more substantial here that takes into
    #       account divergent/erratic performance or intentional timing
    #       clusters that are not evenly dispersed
    na = np.array(wait_seconds)
    average_wait = na.mean()
    return average_wait


def generate_wait_times(trips_and_stop_times: pd.DataFrame
                        ) -> Dict[int, List[float]]:
    wait_times = {0: [], 1: []}
    for stop_id in trips_and_stop_times.stop_id:

        # Handle both inbound and outbound directions
        for direction in [0, 1]:
            constraint_1 = (trips_and_stop_times.direction_id == direction)
            constraint_2 = (trips_and_stop_times.stop_id == stop_id)
            both_constraints = (constraint_1 & constraint_2)
            direction_subset = trips_and_stop_times[both_constraints]

            # Only run if each direction is contained
            # in the same trip id
            if direction_subset.empty:
                average_wait = None
            else:
                average_wait = calculate_average_wait(direction_subset)

            # Add according to which direction we are working with
            wait_times[direction].append(average_wait)

    return wait_times


def generate_all_observed_edge_costs(trips_and_stop_times: pd.DataFrame
                                     ) -> Union[None, pd.DataFrame]:
    all_edge_costs = []
    all_from_stop_ids = []
    all_to_stop_ids = []
    for trip_id in trips_and_stop_times.trip_id.unique():
        tst_mask = (trips_and_stop_times.trip_id == trip_id)
        tst_sub = trips_and_stop_times[tst_mask]

        # Just in case both directions are under the same trip id
        for direction in [0, 1]:
            dir_mask = (tst_sub.direction_id == direction)
            tst_sub_dir = tst_sub[dir_mask]

            tst_sub_dir = tst_sub_dir.sort_values('stop_sequence')
            deps = tst_sub_dir.departure_time[:-1]
            arrs = tst_sub_dir.arrival_time[1:]

            # Use .values to strip existing indices
            edge_costs = np.subtract(arrs.values, deps.values)

            # Add each resulting list to the running array totals
            all_edge_costs += list(edge_costs)

            fr_ids = tst_sub_dir.stop_id[:-1].values
            all_from_stop_ids += list(fr_ids)

            to_ids = tst_sub_dir.stop_id[1:].values
            all_to_stop_ids += list(to_ids)

    # Only return a dataframe if there is contents to populate
    # it with
    if len(all_edge_costs) > 0:
        # Now place results in data frame
        return pd.DataFrame({
            'edge_cost': all_edge_costs,
            'from_stop_id': all_from_stop_ids,
            'to_stop_id': all_to_stop_ids})

    # Otherwise a None value should be returned
    else:
        return None


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


def generate_summary_wait_times(df: pd.DataFrame) -> pd.DataFrame:
    df_sub = df[['stop_id',
                 'wait_dir_0',
                 'wait_dir_1']].reset_index(drop=True)
    init_of_stop_ids = df_sub.stop_id.unique()

    # TODO: Use NaN upstream so we don't have this sort of
    #       hacky typing (floats support NaNs) and None conditioning

    # First convert all None values to NaN so we can handle them
    # in vector format
    dir0_mask = df_sub.wait_dir_0.isnull()
    dir1_mask = df_sub.wait_dir_1.isnull()
    df_sub.loc[dir0_mask, 'wait_dir_0'] = np.nan
    df_sub.loc[dir1_mask, 'wait_dir_1'] = np.nan

    # Convert anything that is 0 or less seconds to a NaN as well
    # as there should not be negative or 0 second waits in the system
    df_sub.loc[~(df_sub.wait_dir_0 > 0), 'wait_dir_0'] = np.nan
    df_sub.loc[~(df_sub.wait_dir_1 > 0), 'wait_dir_1'] = np.nan

    # Convert to type float (which support float)
    df_sub.wait_dir_0 = df_sub.wait_dir_0.astype(float)
    df_sub.wait_dir_1 = df_sub.wait_dir_1.astype(float)

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

    if (len(dir_0_check_2) > 0) or (len(dir_1_check_2) > 0):
        raise Exception('NaN values for both directions on some stop IDs.')

    grouped = df_sub.groupby('stop_id')
    summarized = grouped.apply(summarize_waits_at_one_stop)

    summed_reset = summarized.reset_index(drop=False)
    summed_reset.columns = ['stop_id', 'avg_cost']

    end_of_stop_ids = summed_reset.stop_id.unique()
    log('Original stop id count: {}'.format(len(init_of_stop_ids)))
    log('After cleaning stop id count: {}'.format(len(end_of_stop_ids)))

    # Check for the presence of any unresolved stop ids and
    # assign them some value boarding cost
    if len(init_of_stop_ids) > len(end_of_stop_ids):
        a = set(list(init_of_stop_ids))
        b = set(list(end_of_stop_ids))
        unresolved_ids = list(a - b)
        log('Some unaccounted for stop '
            'ids. Resolving {}...'.format(len(unresolved_ids)))

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
            acst.append(30 * 60)  # 30 minutes, converted to seconds

        # Rebuild the dataframe
        summed_reset = pd.DataFrame({'stop_id': sids, 'avg_cost': acst})

    return summed_reset


def generate_edge_and_wait_values(feed: ptg.gtfs.feed,
                                  target_time_start: int,
                                  target_time_end: int) -> Tuple[pd.DataFrame]:
    all_edge_costs = None
    all_wait_times = None
    for i, route in feed.routes.iterrows():
        log('Processing on route {}.'.format(route.route_id))

        # Get all the subset of trips that are related to this route
        route_match_mask = (feed.trips.route_id == route.route_id)
        trips = feed.trips[route_match_mask]

        # Get just the stop times related to this trip
        st_trip_id_mask = feed.stop_times.trip_id.isin(trips.trip_id)
        stimes_init = feed.stop_times[st_trip_id_mask]

        # Then subset further by just the time period that we care about
        start_time_mask = (stimes_init.arrival_time >= target_time_start)
        end_time_mask = (stimes_init.arrival_time <= target_time_end)
        stimes = stimes_init[start_time_mask & end_time_mask]

        # Let user know how it is going
        # TODO: Make these logger.info statements
        a = len(stimes_init.trip_id.unique())
        b = len(stimes.trip_id.unique())
        log('\tReduced trips in consideration from {} to {}.'.format(a, b))

        trips_and_stop_times = pd.merge(trips,
                                        stimes,
                                        how='inner',
                                        on='trip_id')

        trips_and_stop_times = pd.merge(trips_and_stop_times,
                                        feed.stops,
                                        how='inner',
                                        on='stop_id')

        sort_values_list = ['stop_sequence',
                            'arrival_time',
                            'departure_time']
        trips_and_stop_times = trips_and_stop_times.sort_values(
            sort_values_list)
        trips_and_stop_times = pd.merge(trips,
                                        stimes,
                                        how='inner',
                                        on='trip_id')

        trips_and_stop_times = pd.merge(trips_and_stop_times,
                                        feed.stops,
                                        how='inner',
                                        on='stop_id')

        sort_values_list = ['stop_sequence',
                            'arrival_time',
                            'departure_time']
        trips_and_stop_times = trips_and_stop_times.sort_values(
            sort_values_list)

        wait_times = generate_wait_times(trips_and_stop_times)
        trips_and_stop_times['wait_dir_0'] = wait_times[0]
        trips_and_stop_times['wait_dir_1'] = wait_times[1]

        tst_sub = trips_and_stop_times[['stop_id',
                                        'wait_dir_0',
                                        'wait_dir_1']]

        # Add to the running total for wait times in this feed subset
        if all_wait_times is None:
            all_wait_times = tst_sub
        else:
            all_wait_times = all_wait_times.append(tst_sub)

        # Get all edge costs for this route and add to the running total
        edge_costs = generate_all_observed_edge_costs(trips_and_stop_times)

        # Add to the running total in this feed subset
        if all_edge_costs is None:
            all_edge_costs = edge_costs
        else:
            all_edge_costs = all_edge_costs.append(edge_costs)

    return (all_edge_costs, all_wait_times)
