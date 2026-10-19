import itertools

import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import fixed_threshold_flags, moving_average_z_flags, replay_flags
from evaluation.harness import EvalSettings, make_replay_runner, natural_noise, run_series_eval, sample_dates
from evaluation.inject import AnomalyInjector, inject_segment, injected_direction
from evaluation.scoring import (
    Case,
    false_alarms_per_100_days,
    is_match,
    score_attribution,
    score_case,
    summarize,
)

D0 = pd.Timestamp("2016-06-20")


@pytest.fixture(scope="module")
def series() -> pd.Series:
    """Two years of daily data: weekly pattern plus ~2% noise."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2015-01-01", "2016-12-31")
    weekly = 1 + 0.2 * np.sin(2 * np.pi * idx.dayofweek / 7)
    return pd.Series(1000 * weekly * rng.normal(1, 0.02, len(idx)), index=idx)


# ------------------------------------------------------------------ injection


def test_spike_and_drop_change_one_day_only(series):
    inj = AnomalyInjector()
    spiked = inj.inject(series, "spike", D0, 0.5)
    assert spiked[D0] == pytest.approx(series[D0] * 1.5)
    assert (spiked.drop(D0) == series.drop(D0)).all()
    assert inj.inject(series, "drop", D0, 0.3)[D0] == pytest.approx(series[D0] * 0.7)


def test_level_shift_persists(series):
    shifted = AnomalyInjector().inject(series, "level_shift", D0, 0.2)
    assert (shifted[:D0].iloc[:-1] == series[:D0].iloc[:-1]).all()
    assert (shifted[D0:] / series[D0:]).tolist() == pytest.approx([1.2] * len(series[D0:]))


def test_gradual_drift_ramps_then_holds(series):
    drifted = AnomalyInjector(ramp_days=10).inject(series, "gradual_drift", D0, 0.5)
    ratio = (drifted / series)[D0:]
    assert ratio.iloc[0] == pytest.approx(1.05)
    assert ratio.iloc[9] == pytest.approx(1.5)
    assert ratio.iloc[10:].tolist() == pytest.approx([1.5] * len(ratio.iloc[10:]))


def test_unknown_type_raises_and_missing_date_is_ignored(series):
    with pytest.raises(ValueError):
        AnomalyInjector().inject(series, "bogus", D0, 0.1)
    assert (AnomalyInjector().inject(series, "spike", pd.Timestamp("1999-01-01"), 0.5) == series).all()


def test_injected_direction():
    assert injected_direction("spike", 0.2) == "up"
    assert injected_direction("drop", 0.2) == "down"
    assert injected_direction("level_shift", -0.2) == "down"
    assert injected_direction("drop", -0.2) == "up"


def test_inject_segment_changes_only_that_segment_and_day():
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2016-06-20"] * 4 + ["2016-06-21"] * 2),
            "market": ["A", "A", "B", "B", "A", "B"],
            "sales": [10.0] * 6,
        }
    )
    out, n = inject_segment(df, "d", "sales", "market", "A", D0, 1.0)
    assert n == 2
    assert out["sales"].tolist() == [20.0, 20.0, 10.0, 10.0, 10.0, 10.0]
    assert df["sales"].tolist() == [10.0] * 6  # input untouched


# ------------------------------------------------------------------ baselines


def test_fixed_threshold_flags_big_move_not_small(series):
    inj = AnomalyInjector()
    assert fixed_threshold_flags(inj.inject(series, "spike", D0, 0.5)).get(D0) == "up"
    assert D0 not in fixed_threshold_flags(inj.inject(series, "spike", D0, 0.05))


def test_moving_average_z_flags_spike_and_drop(series):
    inj = AnomalyInjector()
    assert moving_average_z_flags(inj.inject(series, "spike", D0, 1.0)).get(D0) == "up"
    assert moving_average_z_flags(inj.inject(series, "drop", D0, 0.8)).get(D0) == "down"


def test_replay_flags_reports_only_the_day_itself():
    days = list(pd.date_range("2016-06-20", periods=3))
    out = replay_flags({days[1]: "up"}, days)
    assert out[days[0]] == [] and out[days[1]] == [(days[1], "up")]


# -------------------------------------------------------------------- scoring


def test_match_rules():
    spike = Case("spike", 0.5, D0, "up")
    assert is_match(spike, D0, "up", 7)
    assert not is_match(spike, D0 + pd.Timedelta(days=1), "up", 7)  # spike must land on its day
    assert not is_match(spike, D0, "down", 7)  # wrong direction
    shift = Case("level_shift", 0.2, D0, "up")
    assert is_match(shift, D0 + pd.Timedelta(days=3), "up", 7)
    assert not is_match(shift, D0 - pd.Timedelta(days=1), "up", 7)
    assert not is_match(shift, D0 + pd.Timedelta(days=8), "up", 7)


def test_score_case_delay_and_false_alerts():
    case = Case("level_shift", 0.2, D0, "up")
    day = pd.Timedelta(days=1)
    replay = {
        D0: [],
        D0 + day: [],
        D0 + 2 * day: [(D0 + day, "up")],  # first seen two days after the shift began
        D0 + 3 * day: [(D0 + day, "up"), (D0 + 3 * day, "down")],  # wrong direction -> false alert
    }
    res = score_case(case, replay, clean_flags=set(), max_delay=7)
    assert res.detected and res.delay_days == 2 and res.tp == 1 and res.fp == 1


def test_flags_already_on_clean_data_are_not_false_alerts():
    case = Case("spike", 0.5, D0, "up")
    natural = (D0 + pd.Timedelta(days=3), "up")
    replay = {D0: [(D0, "up"), natural]}
    res = score_case(case, replay, clean_flags={natural}, max_delay=7)
    assert res.tp == 1 and res.fp == 0


def test_summarize_and_f1():
    case = Case("spike", 0.5, D0, "up")
    hit = score_case(case, {D0: [(D0, "up")]}, set(), 7)
    miss = score_case(case, {D0: []}, set(), 7)
    s = summarize([hit, miss])
    assert s["recall"] == 0.5 and s["precision"] == 1.0
    assert s["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert s["median_delay"] == 0
    assert summarize([miss])["f1"] == 0.0


def test_false_alarm_rate():
    flags = {(D0, "up"), (D0 + pd.Timedelta(days=5), "down")}
    assert false_alarms_per_100_days(flags, 200) == pytest.approx(1.0)


def test_score_attribution():
    result = {
        "contributors": [{"segment": "market=B"}],
        "by_dimension": {"market": {"contributors": [{"segment": "market=B"}]}, "channel": {"contributors": []}},
    }
    assert score_attribution(result, "market", "B") == (True, True)
    assert score_attribution(result, "market", "A") == (False, False)
    assert score_attribution({}, "market", "B") == (False, False)


# -------------------------------------------------------------------- harness


def simple_detect(series, as_of):
    """Stand-in detector: flag any of the last 3 days that is 25% off the previous 28-day median."""
    ref = series.shift(1).rolling(28, min_periods=14).median()
    dev = ((series - ref) / ref).iloc[-3:]
    return [(d, "up" if v > 0 else "down") for d, v in dev.items() if abs(v) >= 0.25]


def test_sample_dates_respect_gap_and_avoid_list():
    cands = list(pd.date_range("2016-01-01", periods=120))
    avoid = [pd.Timestamp("2016-02-01")]
    rng = np.random.default_rng(0)
    dates = sample_dates(cands, avoid, 4, max_delay=7, min_gap=14, rng=rng)
    assert len(dates) == 4
    assert all(abs((d - avoid[0]).days) > 7 for d in dates)
    assert all((b - a).days >= 14 for a, b in itertools.pairwise(dates))


def test_harness_end_to_end_with_stand_in_detector(series):
    settings = EvalSettings(
        types=("spike", "level_shift"), magnitudes=(0.5,), n_dates=3, eval_days=120, max_delay=5, seed=1
    )
    runners = {
        "detector": make_replay_runner(simple_detect),
        "fixed_threshold": lambda s, dates: replay_flags(fixed_threshold_flags(s), dates),
    }
    out = run_series_eval(series, runners, settings)

    assert out["n_clean_days"] == 121 and len(out["dates"]) == 3
    assert set(out["results"]) == {"detector", "fixed_threshold"}
    assert len(out["results"]["detector"]) == 2 * 1 * 3
    det = summarize(out["results"]["detector"])
    assert det["recall"] == 1.0  # a 50% change is obvious to a 25% rule
    assert det["median_delay"] == 0
    assert out["false_alarms_per_100d"]["detector"] == 0.0  # 2% noise never reaches 25%


def test_harness_fails_loudly_when_no_clean_stretch_exists(series):
    def noisy(series, as_of):
        return [(d, "up") for d in series.index[-3:]]  # flags every day

    with pytest.raises(ValueError, match="No usable injection dates"):
        run_series_eval(series, {"detector": make_replay_runner(noisy)}, EvalSettings(eval_days=60, n_dates=2))


def test_natural_noise_is_close_to_the_injected_noise_level(series):
    # the fixture has ~2% multiplicative noise; the same-weekday median reference adds a little more
    assert 0.015 < natural_noise(series) < 0.04
    assert natural_noise(series.iloc[:5]) is None
