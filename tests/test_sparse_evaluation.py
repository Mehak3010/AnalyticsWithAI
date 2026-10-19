import numpy as np
import pandas as pd

from evaluation.harness import EvalSettings, run_series_eval
from evaluation.inject import AnomalyInjector


def test_evaluation_injects_only_on_observed_sparse_series_dates():
    dates = pd.bdate_range("2025-01-01", periods=100)
    series = pd.Series(100.0, index=dates)

    def no_alerts(values, as_of_dates):
        return {date: [] for date in as_of_dates}

    settings = EvalSettings(
        types=("spike",),
        magnitudes=(0.5,),
        n_dates=3,
        max_delay=3,
        eval_days=60,
        min_gap_days=5,
        seed=7,
    )
    result = run_series_eval(series, {"detector": no_alerts}, settings)

    assert len(result["dates"]) == 3
    assert all(date in series.index for date in result["dates"])

    injector = AnomalyInjector(seed=7)
    for date in result["dates"]:
        injected = injector.inject(series, "spike", date, 0.5)
        assert injected.loc[date] == 150.0
        assert np.count_nonzero(injected.to_numpy() != series.to_numpy()) == 1
