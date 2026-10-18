from anomaly_agent.narration.validator import extract_numbers, validate_numbers

BUNDLE = {
    "metric": "profit_margin",
    "segment": "market=Europe",
    "date": "2017-03-12",
    "observed": 0.104,
    "expected": 0.121,
    "rel_deviation": -0.1405,
    "sales": 1_250_000,
    "drivers": [{"segment": "Europe", "share": 0.7}],
    "late_rate_change_pts": 6.0,
}


def test_extract_handles_commas_decimals_percent_suffix():
    vals = [n.value for n in extract_numbers("Sales were $1,250,000, up 9.4% or 2.5k units")]
    assert vals == [1_250_000.0, 9.4, 2500.0]


def test_valid_text_passes():
    text = "Margin fell from 12.1% to 10.4%. About 70% came from Europe, up 6 points on 12 Mar 2017."
    assert validate_numbers(text, BUNDLE).ok


def test_rounding_is_tolerated():
    assert validate_numbers("Margin fell about 14% versus expected.", BUNDLE).ok
    assert validate_numbers("Sales were about 1.25 million.", BUNDLE).ok


def test_invented_number_fails():
    res = validate_numbers("Margin fell to 8.0% and 70% came from Europe.", BUNDLE)
    assert not res.ok
    assert res.unmatched == ["8.0%"]


def test_precision_matters():
    assert not validate_numbers("Margin was 10.9%.", BUNDLE).ok


def test_sign_is_ignored_for_direction_words():
    assert validate_numbers("Deviation was 14.1% below expected.", BUNDLE).ok


def test_no_numbers_is_ok():
    assert validate_numbers("Nothing unusual to report.", BUNDLE).ok


def test_extra_allowed():
    assert validate_numbers("Check 3 segments.", BUNDLE, extra_allowed=[3]).ok
