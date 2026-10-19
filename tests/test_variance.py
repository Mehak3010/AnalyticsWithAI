import pytest

from anomaly_agent.variance import VarianceEngine


def test_volume_price_mix_bridge_reconciles():
    engine = VarianceEngine.__new__(VarianceEngine)
    result = engine.calculate_bridge(
        {"A": {"volume": 120, "price": 11}, "B": {"volume": 60, "price": 18}},
        {"A": {"volume": 100, "price": 10}, "B": {"volume": 50, "price": 20}},
    )

    assert result["baseline_revenue"] == pytest.approx(2000)
    assert result["actual_revenue"] == pytest.approx(2400)
    assert result["variance"] == pytest.approx(400)
    assert result["bridge"]["total"] == pytest.approx(400)
    assert result["bridge"]["volume"] + result["bridge"]["mix"] + result["bridge"]["price"] == pytest.approx(400)


def test_bridge_handles_new_and_disappearing_segments():
    engine = VarianceEngine.__new__(VarianceEngine)
    result = engine.calculate_bridge(
        {"new": {"volume": 10, "price": 5}},
        {"old": {"volume": 10, "price": 4}},
    )
    assert result["baseline_revenue"] == pytest.approx(40)
    assert result["actual_revenue"] == pytest.approx(50)
    assert result["bridge"]["total"] == pytest.approx(10)


def test_bridge_rejects_negative_volume():
    engine = VarianceEngine.__new__(VarianceEngine)
    with pytest.raises(ValueError, match="non-negative"):
        engine.calculate_bridge({"A": {"volume": -1, "price": 10}}, {"A": {"volume": 1, "price": 10}})


def test_bridge_requires_volume_and_price():
    engine = VarianceEngine.__new__(VarianceEngine)
    with pytest.raises(ValueError, match="volume"):
        engine.calculate_bridge({"A": {"volume": 1}}, {"A": {"volume": 1, "price": 10}})
