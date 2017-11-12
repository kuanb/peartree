import networkx as nx
import numpy as np
import pandas as pd
import random

from utilities import log


def calculate_average_wait(direction_times):
    first = direction_times.arrival_time[1:].values
    second = direction_times.arrival_time[:-1].values
    wait_seconds = (first - second)
    average_wait = np.array(wait_seconds).mean()
    return average_wait


def generate_wait_times(trips_and_stop_times: pd.DataFrame):
    wait_times = {0: [], 1: []}
    for stop_id in trips_and_stop_times.stop_id:

        # Handle both inbound and outbound directions
        for direction in [0, 1]:
            constraint_1 = (trips_and_stop_times.direction_id==direction)
            constraint_2 = (trips_and_stop_times.stop_id==stop_id)
            direction_subset = trips_and_stop_times[constraint_1 & constraint_2]

            # Only run if each direction is contained
            # in the same trip id
            if direction_subset.empty:
                average_wait = None
            else:
                average_wait = calculate_average_wait(direction_subset)

            # Add according to which direction we are working with
            wait_times[direction].append(average_wait)
            
    return wait_times


def generate_all_observed_edge_costs(trips_and_stop_times):
    all_edge_costs = None
    for trip_id in trips_and_stop_times.trip_id.unique():
        tst_mask = (trips_and_stop_times.trip_id==trip_id)
        tst_sub = trips_and_stop_times[tst_mask]

        # Just in case both directions are under the same trip id
        for direction in [0, 1]:
            tst_sub_dir = tst_sub[tst_sub.direction_id==direction]

            tst_sub_dir = tst_sub_dir.sort_values('stop_sequence')
            deps = tst_sub_dir.departure_time[:-1]
            arrs = tst_sub_dir.arrival_time[1:]

            # Use .values to strip existing indices
            edge_costs = np.subtract(arrs.values, deps.values)

            # Now place results in data frame
            new_edges = pd.DataFrame({'edge_cost': edge_costs})
            new_edges['from_stop_id'] = tst_sub_dir.stop_id[:-1].values
            new_edges['to_stop_id'] = tst_sub_dir.stop_id[1:].values

            if all_edge_costs is None:
                all_edge_costs = new_edges
            else:
                all_edge_costs = all_edge_costs.append(new_edges)
    return all_edge_costs


def summarize_edge_costs(df):
    from_stop_id = df.from_stop_id.values[0]
    results_mtx = []
    for to_stop_id in df.to_stop_id.unique():
        to_mask = (df.to_stop_id==to_stop_id)
        avg_cost = df[to_mask].edge_cost.mean()
        results_mtx.append([avg_cost,
                            from_stop_id,
                            to_stop_id])
    return pd.DataFrame(results_mtx, columns=df.columns)


def generate_summary_edge_costs(all_edge_costs):
    summary_groupings = all_edge_costs.groupby('from_stop_id')
    summary = summary_groupings.apply(_summarize_edge_costs)
    summary = summary.reset_index(drop=True)
    return summary


def summarize_waits_at_one_stop(stop_df):
    divide_by = len(stop_df) * 2
    dir_0_sum = stop_df.wait_dir_0.sum()
    dir_1_sum = stop_df.wait_dir_1.sum()
    calculated = ((dir_0_sum + dir_1_sum)/divide_by)
    
    return calculated


def generate_summary_wait_times(df):
    df_sub = df[['stop_id',
                 'wait_dir_0',
                 'wait_dir_1']].reset_index(drop=True)
    init_of_stop_ids = df_sub.stop_id.unique()
    
    # TODO: Use NaN upstream so we don't have this sort of
    #       hacky typing (floats support NaNs) and None conditioning
    
    # First convert all None values to NaN so we can handle them
    # in vector format
    df_sub.loc[df_sub.wait_dir_0==None, 'wait_dir_0'] = np.nan
    df_sub.loc[df_sub.wait_dir_1==None, 'wait_dir_1'] = np.nan
    
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
        log('\ndir_0_check_1 values')
        log(dir_0_check_2.head())
        log('\ndir_1_check_2 values')
        log(dir_1_check_2.head())
        raise Exception('NaN values for both directions on some stop IDs.')
    
    grouped = df_sub.groupby('stop_id')
    summarized = grouped.apply(_summarize_waits_at_one_stop)
    
    summed_reset = summarized.reset_index(drop=False)
    summed_reset.columns = ['stop_id', 'avg_cost']
    
    end_of_stop_ids = summed_reset.stop_id.unique()
    log(f'Original stop id count: {len(init_of_stop_ids)}')
    log(f'After cleaning stop id count: {len(end_of_stop_ids)}')
    
    if len(init_of_stop_ids) > len(end_of_stop_ids):
        a = set(list(init_of_stop_ids))
        b = set(list(end_of_stop_ids))
        unresolved_ids = list(a - b)
        log('Some unaccounted for stop '
              'ids. Resolving {}...'.format(len(unresolved_ids)))
        
        # TODO: Perhaps these are start/end stops and should adopt
        #       a cost that is "average" for that route?
        # We should think of how to actually do this
        # because we do not have enough data, for now let's
        # just assign some default high cost connection value
        # to these stops
        sids = list(summed_reset.stop_id)
        acst = list(summed_reset.avg_cost)
        for i in unresolved_ids:
            sids.append(i)
            acst.append(30 * 60)  # 30 minutes, converted to seconds
        
        # Rebuild the dataframe
        summed_reset = pd.DataFrame({'stop_id': sids, 'avg_cost': acst})
    
    return summed_reset


def generate_edge_and_wait_values(feed,
                                   target_time_start,
                                   target_time_end):
    all_edge_costs = None
    all_wait_times = None
    for i, route in feed.routes.iterrows():
        log(f'Processing on route {route.route_id}.')
        # Now get all the trips for that route
        trips = feed.trips[feed.trips.route_id==route.route_id]

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
        log('\tReduced trips in consideration from {} to {}.'.format(a,b))

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
        trips_and_stop_times = trips_and_stop_times.sort_values(sort_values_list)
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
        trips_and_stop_times = trips_and_stop_times.sort_values(sort_values_list)

        wait_times = generate_wait_times(trips_and_stop_times)
        trips_and_stop_times['wait_dir_0'] = wait_times[0]
        trips_and_stop_times['wait_dir_1'] = wait_times[1]

        tst_sub = trips_and_stop_times[['stop_id',
                                        'wait_dir_0',
                                        'wait_dir_1']]

        if all_wait_times is None:
            all_wait_times = tst_sub
        else:
            all_wait_times = all_wait_times.append(tst_sub)    

        # Get all edge costs for this route and add to the running total
        edge_costs = generate_all_observed_edge_costs(trips_and_stop_times)

        # Add to the running total
        if all_edge_costs is None:
            all_edge_costs = edge_costs
        else:
            all_edge_costs = all_edge_costs.append(edge_costs)
            
    return (all_edge_costs, all_wait_times)


def generate_summary_graph_elements(feed, target_time_start, target_time_end):
    (all_edge_costs,
     all_wait_times) = generate_edge_and_wait_values(feed,
                                                      target_time_start,
                                                      target_time_end)
    
    summary_edge_costs = generate_summary_edge_costs(all_edge_costs)
    wait_times_by_stop = generate_summary_wait_times(all_wait_times)
    
    return (summary_edge_costs, wait_times_by_stop)


def generate_random_name():
    choices = (string.ascii_uppercase + string.digits)
    return ''.join(random.SystemRandom().choice(choices) for _ in range(N))
