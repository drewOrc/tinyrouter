from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from tinyrouter import figures  # noqa: E402

EXPECTED = {"learning_curves.png", "risk_coverage.png", "routers.png", "threshold_transfer.png"}


def test_figures_are_drawn_from_json_and_are_byte_identical_on_rerun(tmp_path):
    first = figures.build(Path("results"), tmp_path / "a")
    second = figures.build(Path("results"), tmp_path / "b")
    assert {p.name for p in first} == EXPECTED
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
        assert b"Matplotlib version" not in a.read_bytes()


def test_risk_coverage_refuses_a_point_whose_seeds_chose_different_aggregations():
    summary = {"groups": {"x": {"result": {"selected_aggregation": ["argmax", "summed"]}}}}
    with pytest.raises(ValueError, match="different aggregations"):
        figures.selected_aggregation(summary, "x")


def test_threshold_transfer_skips_points_without_a_feasible_threshold_on_every_seed():
    summary = figures.load(Path("results"), "analysis/summary.json")
    series = figures.transfer_series(summary, "modernbert")
    ks = [k for k, _, _ in series["test"]]
    assert 1 not in ks
    assert ks == sorted(ks)
    assert series["test"][-1][0] == 100
