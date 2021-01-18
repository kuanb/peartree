import numpy as np
import pandas as pd
import pytest

from peartree.parallel import generate_wait_times


@pytest.mark.parametrize(
    "trips_and_stop_times, expected",
    [
        # missing direction id in GTFS, wait times the same both ways
        (
            pd.DataFrame({
                "stop_id": ["a", "a"],
                "arrival_time": [1, 2],
            }),
            {0: {"a": 3}, 1: {"a": 3}},
        ),
        # has direction id in GTFS, different wait times each way
        (
            pd.DataFrame({
                "stop_id": ["a", "a", "b"],
                "arrival_time": [1, 2, 10],
                "direction_id": [0, 0, 1],
            }),
            {0: {"a": 3, "b": np.nan}, 1: {"a": np.nan, "b": 10}},
        )
    ])
def test_generate_wait_times(trips_and_stop_times, expected):
    time_start = 1
    time_end = 10
    # simple placeholder cost function for testing
    stop_coster = lambda a, b, c: sum(c)
    res = generate_wait_times(
        time_start,
        time_end,
        trips_and_stop_times,
        stop_coster)
    print("res", res)
    assert res == expected