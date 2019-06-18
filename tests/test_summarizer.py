import numpy as np
import pandas as pd
from peartree.summarizer import generate_summary_wait_times


def test_generate_summary_wait_times():
    df = pd.DataFrame({
        'stop_id': [
            1,
            1,
            2,
            2,
            3,
            4],
        'wait_dir_0': [
            10,
            10,
            19,
            21,
            np.nan,
            12],
        'wait_dir_1': [
            np.nan,
            np.nan,
            9,
            11,
            np.nan,
            np.nan],
    })

    fallback_stop_cost = 40.0  # seconds
    res = generate_summary_wait_times(df, fallback_stop_cost)
    res = res.sort_values(by='stop_id')
    assert res['stop_id'].tolist() == [1, 2, 3, 4]
    assert res['avg_cost'].tolist() == [10.0, 15.0, 40.0, 12.0]
