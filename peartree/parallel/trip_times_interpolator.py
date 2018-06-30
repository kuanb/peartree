from multiprocessing.managers import BaseManager
from typing import List, Union

import numpy as np
import pandas as pd

from peartree.toolkit import nan_helper


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
                col_array = sub_df[col].astype(float)

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
    nans, x = nan_helper(orig_array)
    orig_array[nans] = np.interp(x(nans), x(~nans), orig_array[~nans])
    return orig_array


def make_new_trip_time_interpolator_manager():
    manager = TripTimesInterpolatorManager()
    manager.start()
    return manager


TripTimesInterpolatorManager.register('TripTimesInterpolator', TripTimesInterpolator)
