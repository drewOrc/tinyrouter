import json
from pathlib import Path

import pytest

from tinyrouter import cost
from tinyrouter.cost import break_even_queries, local_inference_usd_per_query, training_cost_usd

RESULTS = Path("results")


def test_training_cost_prices_wall_clock_hours():
    assert training_cost_usd(3600, 2.0) == pytest.approx(2.0)
    assert training_cost_usd(1800, 0.5) == pytest.approx(0.25)


def test_local_inference_cost_scales_with_latency_threads_and_price():
    # 1 s on 4 vCPUs at US$0.05 per vCPU-hour
    assert local_inference_usd_per_query(1000, 4, 0.05) == pytest.approx(4 * 0.05 / 3600)


def test_break_even_divides_one_time_cost_by_the_per_query_saving():
    assert break_even_queries(10.0, 0.001, 0.0005) == pytest.approx(20_000)


@pytest.mark.parametrize("router_cost", [0.001, 0.002])
def test_break_even_is_none_when_the_router_saves_nothing(router_cost):
    assert break_even_queries(10.0, 0.001, router_cost) is None


def test_committed_cost_json_is_what_make_cost_writes():
    committed = json.loads((RESULTS / "cost" / "cost.json").read_text())
    assert cost.build(RESULTS) == committed


def test_measured_and_assumed_numbers_live_in_separate_blocks():
    body = cost.build(RESULTS)
    assert "accelerator_usd_per_hour" in body["assumptions"]
    assert "usd_per_hour" not in json.dumps(body["measured"])
    assert body["assumptions"]["accelerator_usd_per_hour"] == [0.5, 1.0, 2.0]
    assert body["assumptions"]["label_usd_per_example"] == [0.05, 0.2, 1.0]
    assert "not a forecast" in body["break_even"]["label"]


def test_break_even_row_matches_a_hand_computation_from_the_raw_inputs():
    """ModernBERT k=10 hybrid at US$1/h with labels at US$0.2, from the files, not cost.py."""
    runs = [
        json.loads((RESULTS / "runs" / f"ModernBERT-base-k10-seed{s}.json").read_text())
        for s in (42, 43, 44)
    ]
    seconds = sum(r["training"]["train_wall_seconds"] for r in runs) / 3
    rows_labelled = runs[0]["training"]["train_rows"]
    summary = json.loads((RESULTS / "analysis" / "summary.json").read_text())
    hybrid = summary["groups"]["modernbert/k10"]["result"]["final"]["fallback"]["0.02"]
    api = hybrid["hybrid"]["test"]["llm_cost_usd_per_1k"]["mean"] / 1000
    haiku = json.loads((RESULTS / "analysis" / "haiku.json").read_text())
    llm_only = haiku["splits"]["test"]["cost_usd_per_1k_queries"] / 1000
    bench = json.loads((RESULTS / "efficiency" / "cpu_latency.json").read_text())
    p50 = bench["models"]["modernbert"]["end_to_end"]["p50_ms"]
    local = p50 / 1000 / 3600 * bench["method"]["intra_op_threads"] * 0.05
    expected = (seconds / 3600 * 1.0 + 0.2 * rows_labelled) / (llm_only - api - local)

    row = next(
        r
        for r in cost.build(RESULTS)["break_even"]["rows"]
        if (r["point"], r["router"], r["accelerator_usd_per_hour"], r["label_usd_per_example"])
        == ("modernbert/k10", "hybrid", 1.0, 0.2)
    )
    assert row["break_even_queries"] == pytest.approx(expected, abs=0.1)
    assert rows_labelled == 1525


def test_every_break_even_scenario_is_present_once():
    rows = cost.build(RESULTS)["break_even"]["rows"]
    keys = {
        (r["point"], r["router"], r["accelerator_usd_per_hour"], r["label_usd_per_example"])
        for r in rows
    }
    assert len(keys) == len(rows) == 2 * 2 * 3 * 4
