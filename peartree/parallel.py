from multiprocessing.managers import BaseManager
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from peartree.toolkit import nan_helper
from peartree.utilities import log


class RouteProcessorManager(BaseManager):
    pass


class RouteProcessor(object):

    def __init__(
            self,
            target_time_start: int,
            target_time_end: int,
            feed_trips: pd.DataFrame,
            stop_times: pd.DataFrame,
            all_stops: pd.DataFrame):

        # Initialize common parameters
        self.target_time_start = target_time_start
        self.target_time_end = target_time_end
        self.stop_times = stop_times.copy()

        # We use route_id as the index to ensure that subselection by
        # route_id from target_route_ids more performant
        self.trips = feed_trips.copy().set_index('route_id', drop=False)

        # Ensure that stop_ids are cast as string
        astops = all_stops.copy()
        astops['stop_id'] = astops['stop_id'].astype(str)
        self.all_stops = astops

    def generate_route_costs(self, route_id: str):
        # Get all the subset of trips that are related to this route
        trips = self.trips.loc[route_id].copy()

        # Pandas will try and make returned result a Series if there
        # is only one result - prevent this from happening
        if isinstance(trips, pd.Series):
            trips = trips.to_frame().T

        # Get just the stop times related to this trip
        st_trip_id_mask = self.stop_times.trip_id.isin(trips.trip_id)
        stimes_init = self.stop_times[st_trip_id_mask].copy()

        # Then subset further by just the time period that we care about
        start_time_mask = (stimes_init.arrival_time >= self.target_time_start)
        end_time_mask = (stimes_init.arrival_time <= self.target_time_end)
        stimes = stimes_init[start_time_mask & end_time_mask]

        # Report on progress if requested
        a = len(stimes_init.trip_id.unique())
        b = len(stimes.trip_id.unique())
        log('\tReduced selected trips on route {} from {} to {}.'.format(
            route_id, a, b))

        trips_and_stop_times = pd.merge(trips,
                                        stimes,
                                        how='inner',
                                        on='trip_id')

        trips_and_stop_times = pd.merge(trips_and_stop_times,
                                        self.all_stops.copy(),
                                        how='inner',
                                        on='stop_id')

        sort_list = ['stop_sequence',
                     'arrival_time',
                     'departure_time']
        trips_and_stop_times = trips_and_stop_times.sort_values(sort_list)

        # Check direction_id column value before using
        # trips_and_stop_times to generate wait and edge costs
        # Note: Advantage to adding handling at route level is that peartree
        #       avoids tossing direction id if a specific route has all direction
        #       id rows filled in (while another does not, which is possible).
        if 'direction_id' in trips_and_stop_times:
            # If there is such column then check if it contains NaN
            has_nan = trips_and_stop_times['direction_id'].isnull()
            if len(trips_and_stop_times[has_nan]) > 0:
                # If it has no full coverage in direction_id, drop the column
                trips_and_stop_times.drop('direction_id', axis=1, inplace=True)

        wait_times = generate_wait_times(trips_and_stop_times)

        # Look up wait time for each stop in wait_times for each direction
        wait_zero = trips_and_stop_times['stop_id'].apply(lambda x: wait_times[0][x])
        trips_and_stop_times['wait_dir_0'] = wait_zero
        
        wait_one = trips_and_stop_times['stop_id'].apply(lambda x: wait_times[1][x])
        trips_and_stop_times['wait_dir_1'] = wait_one

        tst_sub = trips_and_stop_times[['stop_id',
                                        'wait_dir_0',
                                        'wait_dir_1']]

        # Get all edge costs for this route and add to the running total
        edge_costs = generate_all_observed_edge_costs(trips_and_stop_times)

        return (tst_sub, edge_costs)


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
    average_wait = na.mean() / 2  # half headway
    return average_wait


def generate_wait_times(trips_and_stop_times: pd.DataFrame
                        ) -> Dict[int, List[float]]:
    wait_times = {0: {}, 1: {}}
    for stop_id in trips_and_stop_times.stop_id.unique():
        # Handle both inbound and outbound directions
        for direction in [0, 1]:
            # Check if direction_id exists in source data
            if 'direction_id' in trips_and_stop_times:
                constraint_1 = (trips_and_stop_times.direction_id == direction)
                constraint_2 = (trips_and_stop_times.stop_id == stop_id)
                both_constraints = (constraint_1 & constraint_2)
                direction_subset = trips_and_stop_times[both_constraints]
            else:
                direction_subset = trips_and_stop_times.copy()

            # Only run if each direction is contained
            # in the same trip id
            if direction_subset.empty:
                average_wait = np.nan
            else:
                average_wait = calculate_average_wait(direction_subset)

            # Add according to which direction we are working with
            wait_times[direction][stop_id] = average_wait

    return wait_times


def generate_all_observed_edge_costs(trips_and_stop_times: pd.DataFrame
                                     ) -> Union[None, pd.DataFrame]:
    # TODO: This edge case should be handled up stream. If there is
    #       no direction id upstream, when the trip and stop times
    #       dataframe is created, then it should be added there and all
    #       directions should be set to default 0 or 1.
    # Make sure that the GTFS feed has a direction id
    has_dir_col = 'direction_id' in trips_and_stop_times.columns.values

    all_edge_costs = []
    all_from_stop_ids = []
    all_to_stop_ids = []
    for trip_id in trips_and_stop_times.trip_id.unique():
        tst_mask = (trips_and_stop_times.trip_id == trip_id)
        tst_sub = trips_and_stop_times[tst_mask]

        # Just in case both directions are under the same trip id
        for direction in [0, 1]:
            # Support situations wheredirection_id is absent from the
            # GTFS data. In such situations, include all trip and stop
            # time data, instead of trying to split on that column
            # (since it would not exist).
            if has_dir_col:
                dir_mask = (tst_sub.direction_id == direction)
                tst_sub_dir = tst_sub[dir_mask]
            else:
                tst_sub_dir = tst_sub.copy()

            tst_sub_dir = tst_sub_dir.sort_values('stop_sequence')
            deps = tst_sub_dir.departure_time[:-1]
            arrs = tst_sub_dir.arrival_time[1:]

            # Use .values to strip existing indices
            edge_costs = np.subtract(arrs.values, deps.values)

            # TODO(kuanb): Negative values can result here!
            # HACK: There are times when the arrival and departure data
            #       are "out of order" which results in negative values.
            #       From the values I've looked at, these are edge cases
            #       that have to do with start/end overlaps. I don't have
            #       a good answer for dealing with these but, since they
            #       are possible noise, they can be override by taking
            #       their absolute value.
            edge_costs = np.absolute(edge_costs)

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


def make_route_processor_manager():
    manager = RouteProcessorManager()
    manager.start()
    return manager


class NonUniqueSequenceSet(Exception):
    pass


class TripTimesInterpolatorManager(BaseManager):
    pass


class TripTimesInterpolator(object):

    def __init__(
            self,
            stop_times_original_df: pd.DataFrame):

        # Initialize common parameters
        stop_times = stop_times_original_df.copy()

        # Set index on trip id so we can quicly subset the dataframe
        # during iteration of generate_infilled_times
        stop_times = stop_times.set_index('trip_id')

        # Also avoid having these be object column types
        for col in ['arrival_time', 'departure_time']:
            stop_times[col] = stop_times[col].astype(float)

        # Now we can set to self
        self.stop_times = stop_times

    def generate_infilled_times(self, trip_id: str):
        # Get all the subset of trips that are related to this route
        sub_df = self.stop_times.loc[trip_id].copy()

        # Pandas will try and make returned result a Series if there
        # is only one result - prevent this from happening
        if isinstance(sub_df, pd.Series):
            sub_df = sub_df.to_frame().T

            # We again want to make sure these columns are
            # typed right and the pivot itself will just leave
            # them as object type columns, which will cause errors
            # when we check the row for NaN values later on
            for col in ['arrival_time', 'departure_time']:
                sub_df[col] = sub_df[col].astype(float)

        # TODO: Should we be able to assume that this column is
        #   present by the time we arrive here? If so, we should
        #   be able to move this check upstream, earlier in tool

        # Note: Make sure that there is a set of stop sequence
        #       numbers present in each of the trip_id sub-dataframes
        if 'stop_sequence' not in sub_df.columns:
            sub_df['stop_sequence'] = range(len(sub_df))

        uniq_sequence_ids = sub_df.stop_sequence.unique()
        if not len(uniq_sequence_ids) == len(sub_df):
            raise NonUniqueSequenceSet(
                'Expected there to be a unique set of '
                'stop ids for each trip_id in stop_times.')

        # Next, make sure that the subset dataframe is sorted
        # stop sequence, incrementing upward
        sub_df = sub_df.sort_values(by=['stop_sequence'])

        # Extract the arrival and departure times as independent arrays
        for col in ['arrival_time', 'departure_time']:
            sub_df[col] = apply_interpolation(sub_df[col])

        # Re-add the trip_id as column at this point
        sub_df['trip_id'] = trip_id

        # Also, we dump any index set on this subset to avoid issues
        # when returned later
        sub_df = sub_df.reset_index(drop=True)

        # Now free to release/return
        return sub_df


def apply_interpolation(orig_array: List) -> List:
    target_col_array = orig_array.copy()
    nans, x = nan_helper(target_col_array)
    target_col_array[nans] = np.interp(x(nans),
                                       x(~nans),
                                       target_col_array[~nans])
    return target_col_array


def make_trip_time_interpolator_manager():
    manager = TripTimesInterpolatorManager()
    manager.start()
    return manager


TripTimesInterpolatorManager.register(
    'TripTimesInterpolator', TripTimesInterpolator)

RouteProcessorManager.register('RouteProcessor', RouteProcessor)
