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


def test_threshold_transfer_drops_a_point_where_only_some_seeds_were_feasible():
    """BERT k=1 has a feasible 2% threshold on one seed of three; a mean of one is not plotted."""
    summary = figures.load(Path("results"), "analysis/summary.json")
    stat = summary["groups"]["bert/k1"]["result"]["diagnostics"]["0.02"]["test"]["selective_risk"]
    assert stat["n"] == 1
    for name, points in figures.transfer_series(summary, "bert").items():
        assert 1 not in [k for k, _, _ in points], name


def test_router_accuracy_panel_uses_points_not_truncated_bars():
    """A bar starting at 75% would turn 91.9 vs 92.1 into a visible length gap (review R3)."""
    plt = figures.pyplot()
    summary = figures.load(Path("results"), "analysis/summary.json")
    fig = figures.routers(summary, plt)
    accuracy, calls = fig.axes[:2]
    assert len(accuracy.patches) == 0
    assert "points" in accuracy.get_title()
    assert len(calls.patches) > 0
    assert calls.get_ylim()[0] == 0
    plt.close(fig)
