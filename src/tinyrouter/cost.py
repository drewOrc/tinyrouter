"""RQ5 cost: measured numbers and assumed prices, kept apart (docs/PLAN.md section 3, RQ5).

``make cost`` reads committed JSON only and writes ``results/cost/cost.json``:

- ``measured``: M4 (MPS) training wall-clock per encoder and k, CPU latency
  (``results/efficiency/cpu_latency.json``), Haiku's real tokens and spend
  (``results/llm/haiku-8way.json``, ``results/analysis/haiku.json``) and each
  router's real Haiku spend on test (``results/analysis/summary.json``).
- ``assumptions``: prices nobody paid. Training is priced as if the measured
  M4 wall-clock hours were billed at a few accelerator-hour rates; the Mac's
  purchase price is not used. Labelling cost is a sensitivity only (CLINC150
  is an existing dataset). Local inference is priced per vCPU-hour.
- ``break_even``: queries after which a one-time cost (training, optionally
  labelling) is paid back by the per-query saving against LLM-only:
  ``one_time_usd / (llm_only_per_query - router_per_query)``. A scenario
  sensitivity, not a forecast.

Every number is a mean over seeds 42, 43, 44 where seeds exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ACCELERATOR_USD_PER_HOUR = (0.5, 1.0, 2.0)
LABEL_USD_PER_EXAMPLE = (0.05, 0.2, 1.0)
VCPU_USD_PER_HOUR = 0.05
TARGET = "0.02"
MODELS = ("bert", "modernbert")
KS = (1, 5, 10, 25, 50, 100)
POINTS = (("modernbert", 10), ("modernbert", 100))
ROUTERS = ("small_only", "hybrid")
INPUTS = (
    "analysis/summary.json",
    "analysis/haiku.json",
    "llm/haiku-8way.json",
    "efficiency/cpu_latency.json",
    "curves/bert.json",
    "curves/modernbert.json",
)
ASSUMPTION_NOTES = {
    "training": (
        "measured M4 wall-clock hours of one training run, priced as if billed at each "
        "accelerator-hour rate; these are assumed rates, not money spent (the runs used a Mac "
        "already owned, and its purchase price is not charged to a run)"
    ),
    "labelling": (
        "CLINC150 is an existing dataset, so no labelling was paid; cost = labelled training "
        "rows x an assumed price per label, a sensitivity only"
    ),
    "local_inference": (
        "measured CPU p50 end-to-end latency x intra-op threads (vCPUs) x an assumed price per "
        "vCPU-hour, at full utilisation; idle capacity would cost more"
    ),
}


def training_cost_usd(wall_seconds: float, usd_per_hour: float) -> float:
    return wall_seconds / 3600.0 * usd_per_hour


def local_inference_usd_per_query(latency_ms: float, vcpus: int, usd_per_vcpu_hour: float) -> float:
    return latency_ms / 1000.0 / 3600.0 * vcpus * usd_per_vcpu_hour


def break_even_queries(
    one_time_usd: float, llm_only_per_query: float, router_per_query: float
) -> float | None:
    """Queries until the one-time cost is repaid; None when the router saves nothing per query."""
    saving = llm_only_per_query - router_per_query
    if saving <= 0:
        return None
    return one_time_usd / saving


def mean_std(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else None,
        "values": [round(float(v), 6) for v in values],
    }


def load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def training_runs(root: Path, model: str) -> dict[int, list[dict]]:
    """Each k's run JSON ``training`` blocks, from the curve index (BERT k=100 is AC2)."""
    index = load(root, f"curves/{model}.json")
    runs: dict[int, list[dict]] = {}
    for point in sorted(index["points"], key=lambda p: (p["k"], p["seed"])):
        run = load(root, f"runs/{point['run_name']}.json")
        runs.setdefault(point["k"], []).append(run["training"])
    return runs


def measured_training(root: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for model in MODELS:
        out[model] = {}
        for k, blocks in training_runs(root, model).items():
            devices = {b["device"] for b in blocks}
            out[model][str(k)] = {
                "train_wall_seconds": mean_std([b["train_wall_seconds"] for b in blocks]),
                "train_rows": blocks[0]["train_rows"],
                "steps": blocks[0]["global_step"],
                "device": devices.pop() if len(devices) == 1 else sorted(devices),
            }
    return out


def measured_haiku(root: Path) -> dict[str, object]:
    totals = load(root, "llm/haiku-8way.json")["totals"]
    splits = load(root, "analysis/haiku.json")["splits"]
    calls = totals["calls"]
    return {
        "model": load(root, "llm/haiku-8way.json")["identity"]["model"],
        "calls": calls,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "cost_usd": totals["cost_usd"],
        "input_tokens_per_1k_queries": round(1000 * totals["input_tokens"] / calls, 3),
        "output_tokens_per_1k_queries": round(1000 * totals["output_tokens"] / calls, 3),
        "cost_usd_per_1k_queries": {s: splits[s]["cost_usd_per_1k_queries"] for s in splits},
    }


def router_block(summary: dict, model: str, k: int) -> dict[str, dict[str, object]]:
    final = summary["groups"][f"{model}/k{k}"]["result"]["final"]
    blocks = {"small_only": final["small_only"], "hybrid": final["fallback"][TARGET]["hybrid"]}
    out = {}
    for name, block in blocks.items():
        metrics = block if name == "small_only" else block["test"]
        out[name] = {
            key: metrics[key]["mean"]
            for key in ("accuracy_8", "llm_call_rate", "llm_cost_usd_per_1k")
        }
    return out


def cpu_latency(root: Path) -> dict[str, object]:
    bench = load(root, "efficiency/cpu_latency.json")
    return {
        "intra_op_threads": bench["method"]["intra_op_threads"],
        "hardware": bench["hardware"]["cpu"],
        "end_to_end_ms": {
            m: {q: bench["models"][m]["end_to_end"][q] for q in ("p50_ms", "p95_ms", "mean_ms")}
            for m in MODELS
        },
    }


def break_even_rows(measured: dict[str, object]) -> list[dict[str, object]]:
    llm_only = measured["haiku"]["cost_usd_per_1k_queries"]["test"] / 1000.0  # type: ignore[index]
    latency = measured["cpu_latency"]  # type: ignore[assignment]
    rows = []
    for model, k in POINTS:
        training = measured["training"][model][str(k)]  # type: ignore[index]
        seconds = training["train_wall_seconds"]["mean"]
        local = local_inference_usd_per_query(
            latency["end_to_end_ms"][model]["p50_ms"],
            latency["intra_op_threads"],
            VCPU_USD_PER_HOUR,
        )
        for router in ROUTERS:
            api = measured["routers"][f"{model}/k{k}"][router]["llm_cost_usd_per_1k"] / 1000.0  # type: ignore[index]
            for price in ACCELERATOR_USD_PER_HOUR:
                for label in (0.0, *LABEL_USD_PER_EXAMPLE):
                    one_time = training_cost_usd(seconds, price) + label * training["train_rows"]
                    rows.append(
                        {
                            "point": f"{model}/k{k}",
                            "router": router,
                            "accelerator_usd_per_hour": price,
                            "label_usd_per_example": label,
                            "one_time_usd": round(one_time, 6),
                            "llm_only_usd_per_query": round(llm_only, 9),
                            "router_api_usd_per_query": round(api, 9),
                            "router_local_usd_per_query": round(local, 9),
                            "break_even_queries": rounded(
                                break_even_queries(one_time, llm_only, api + local)
                            ),
                            "break_even_queries_api_only": rounded(
                                break_even_queries(one_time, llm_only, api)
                            ),
                        }
                    )
    return rows


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def input_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in INPUTS}


def build(root: Path) -> dict[str, object]:
    summary = load(root, "analysis/summary.json")
    measured = {
        "training": measured_training(root),
        "cpu_latency": cpu_latency(root),
        "haiku": measured_haiku(root),
        "routers": {f"{m}/k{k}": router_block(summary, m, k) for m, k in POINTS},
        "router_target_risk": float(TARGET),
    }
    assumptions = {
        "accelerator_usd_per_hour": list(ACCELERATOR_USD_PER_HOUR),
        "label_usd_per_example": list(LABEL_USD_PER_EXAMPLE),
        "vcpu_usd_per_hour": VCPU_USD_PER_HOUR,
        "notes": ASSUMPTION_NOTES,
    }
    return {
        "format_version": 1,
        "inputs_sha256": input_hashes(root),
        "measured": measured,
        "assumptions": assumptions,
        "break_even": {
            "formula": "one_time_usd / (llm_only_usd_per_query - router_usd_per_query); "
            "null when the router saves nothing per query",
            "label": "scenario sensitivity, not a forecast",
            "rows": break_even_rows(measured),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    root = Path(args.results_root)
    out = root / "cost" / "cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(root), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
