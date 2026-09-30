import copy
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from tinyrouter import figures  # noqa: E402

ORIGINAL = json.loads(Path("results/analysis/summary.json").read_text(encoding="utf-8"))
EXPECTED = {"learning_curves.png", "risk_coverage.png", "routers.png", "threshold_transfer.png"}


def test_figures_are_drawn_from_json_and_are_byte_identical_on_rerun(tmp_path):
    first = figures.build(Path("results"), tmp_path / "a")
    second = figures.build(Path("results"), tmp_path / "b")
    assert {p.name for p in first} == EXPECTED
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
        assert b"Matplotlib version" not in a.read_bytes()


def mixed(summary: dict, group: str, per_seed: list[str]) -> dict:
    """``summary`` with ``group``'s seeds choosing ``per_seed``, as analysis writes it."""
    out = copy.deepcopy(summary)
    out["groups"][group]["result"]["selected_aggregation"] = per_seed
    return out


def test_mixed_per_seed_aggregations_get_one_panel_each_titled_with_their_seeds():
    """AC1b attempt 2: ModernBERT k=100 chose argmax, summed, summed; figures crashed."""
    summary = mixed(ORIGINAL, "modernbert/k100", ["argmax", "summed", "summed"])
    assert figures.risk_coverage_panels(summary) == [
        (10, "summed", None),
        (100, "argmax", (42,)),
        (100, "summed", (43, 44)),
    ]
    plt = figures.pyplot()
    fig = figures.risk_coverage(summary, figures.load(Path("results"), "analysis/curves.json"), plt)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        "(a) ModernBERT k=10 (summed)",
        "(b) ModernBERT k=100 (argmax)\nmean of 3 seeds; final router of seed 42",
        "(c) ModernBERT k=100 (summed)\nmean of 3 seeds; final router of seeds 43, 44",
    ]
    plt.close(fig)


def test_unanimous_seeds_keep_the_original_two_panels():
    assert figures.risk_coverage_panels(ORIGINAL) == [(10, "summed", None), (100, "argmax", None)]


def test_figures_build_on_a_summary_with_mixed_aggregations(tmp_path):
    root = tmp_path / "results"
    (root / "analysis").mkdir(parents=True)
    summary = mixed(ORIGINAL, "modernbert/k100", ["argmax", "summed", "summed"])
    summary = mixed(summary, "modernbert/k10", ["summed", "argmax", "summed"])
    (root / "analysis" / "summary.json").write_text(json.dumps(summary))
    shutil.copy(Path("results/analysis/curves.json"), root / "analysis" / "curves.json")
    written = figures.build(root, root / "figures")
    assert {p.name for p in written} == EXPECTED


def test_an_unknown_aggregation_is_refused():
    summary = mixed(ORIGINAL, "modernbert/k100", ["argmax", "median", "summed"])
    with pytest.raises(ValueError, match="unexpected selected_aggregation"):
        figures.seed_aggregations(summary, "modernbert/k100")


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
