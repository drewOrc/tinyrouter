"""``make analysis``: RQ2 to RQ4 over every archived run, written to ``results/analysis/``.

Inputs (all checked before anything is computed; a missing or changed
input stops the run):

- the four indexes under ``results/curves/`` (BERT and ModernBERT curves,
  the OOS ablation, the baselines), each re-verified with
  ``completeness.verify_index``: exactly the expected points, each once,
  every archive's SHA-256 equal on disk, in the manifest and in the index;
- 75 distinct archives in total (36 encoder points, of which BERT k=100
  are the three AC2 runs, 3 ablation points, 36 baseline points), a
  literal here;
- ``results/llm/haiku-8way.jsonl``, re-verified with ``llm_run.verify``
  (8,600 rows, identity and SHA-256), a literal here.

Cross-checks: within each split every archive holds the same gold labels,
and Haiku's ``gold_intent`` equals them row by row.

Outputs (``results/analysis/``, committed): ``summary.json`` (every
scalar, grouped by model and k, mean and sample std over seeds 42/43/44
with the per-seed values), ``curves.json`` (risk-coverage and operating
curves, mean and std), ``haiku.json`` (the LLM alone, both parsers).
Each is written to a temporary file, read back and checked (the expected
groups, three seeds each, both Haiku splits) before it is moved into
place; only then is ``completed analysis (...)`` printed. There is no
randomness anywhere: no bootstrap, no sampling, no random tie-breaking.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyrouter.analysis import (
    AGGREGATIONS,
    SIGNALS,
    SIGNALS_BY_KIND,
    TARGET_RISKS,
    analyze_run,
)
from tinyrouter.archive import MANIFEST_NAME, load_logits
from tinyrouter.completeness import (
    ABLATION_POINTS,
    BASELINE_KEY,
    BASELINE_POINTS,
    CURVE_POINTS,
    ENCODER_KEY,
    IncompleteError,
    verify_index,
)
from tinyrouter.data import sha256_of
from tinyrouter.haiku import HaikuSplit, check_gold, read_records, split_arrays, summarize_split
from tinyrouter.labels import load_label_space
from tinyrouter.llm_run import EXPECTED_ROWS, full_target
from tinyrouter.llm_run import verify as verify_llm
from tinyrouter.selective import COVERAGE_GRID, ONE_SIDED_95_Z

EXPECTED_ARCHIVES = 75
EXPECTED_LLM_ROWS = 8600
EXPECTED_SEEDS = (42, 43, 44)
FORMAT_VERSION = 1
BASELINE_KINDS = {"majority": "majority", "tfidf-centroid": "tfidf"}

Log = Callable[[str], None]


@dataclass(frozen=True)
class Point:
    group: str
    model: str
    k: int
    oos_train: int | None
    seed: int
    kind: str
    run_name: str
    logits_file: str
    logits_sha256: str


@dataclass(frozen=True)
class IndexSpec:
    file: str
    key: tuple[str, ...]
    expected: frozenset


INDEXES = (
    IndexSpec("bert.json", ENCODER_KEY, CURVE_POINTS),
    IndexSpec("modernbert.json", ENCODER_KEY, CURVE_POINTS),
    IndexSpec("oos-ablation.json", ENCODER_KEY, ABLATION_POINTS),
    IndexSpec("baselines.json", BASELINE_KEY, BASELINE_POINTS),
)


def point_from(index_file: str, entry: dict) -> Point:
    if index_file == "baselines.json":
        model, kind, oos_train = entry["baseline"], BASELINE_KINDS[entry["baseline"]], None
    elif index_file == "oos-ablation.json":
        model, kind, oos_train = "modernbert-oos0", "encoder", entry["oos_train"]
    else:
        model, kind, oos_train = index_file.removesuffix(".json"), "encoder", None
    return Point(
        group=f"{model}/k{entry['k']}",
        model=model,
        k=entry["k"],
        oos_train=oos_train,
        seed=entry["seed"],
        kind=kind,
        run_name=entry["run_name"],
        logits_file=entry["logits_file"],
        logits_sha256=entry["logits_sha256"],
    )


def collect_points(results_root: Path) -> list[Point]:
    """Every point of the four indexes, after each index passes its completion check."""
    points = []
    for spec in INDEXES:
        path = results_root / "curves" / spec.file
        verify_index(path, spec.key, spec.expected, results_root, spec.file)
        entries = json.loads(path.read_text(encoding="utf-8"))["points"]
        points.extend(point_from(spec.file, e) for e in entries)
    files = {p.logits_file for p in points}
    if len(points) != EXPECTED_ARCHIVES or len(files) != EXPECTED_ARCHIVES:
        raise IncompleteError(
            f"expected {EXPECTED_ARCHIVES} points on {EXPECTED_ARCHIVES} distinct archives, got "
            f"{len(points)} points on {len(files)} archives"
        )
    return points


def load_haiku(results_root: Path) -> dict[str, HaikuSplit]:
    target = full_target(results_root)
    count = verify_llm(target)
    if count != EXPECTED_LLM_ROWS:
        raise IncompleteError(f"{target.predictions}: {count} rows, expected {EXPECTED_LLM_ROWS}")
    records = read_records(target.predictions)
    return {s: split_arrays(records, s, n) for s, n in EXPECTED_ROWS.items()}


def combine(values: list, keep_values: bool = True) -> object:
    """Seeds merged leaf by leaf: numbers -> mean, sample std (ddof=1) and the values.

    Seeds whose value is None (e.g. no feasible threshold) are left out of
    the mean and counted in ``n``; with fewer than two numbers the std is
    None, not 0.
    """
    if all(v is None for v in values):
        return None
    if all(isinstance(v, dict) for v in values):
        keys = set(values[0])
        if any(set(v) != keys for v in values):
            raise ValueError("seed results have different keys")
        return {k: combine([v[k] for v in values], keep_values) for k in values[0]}
    numbers = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    if numbers and len(numbers) + values.count(None) == len(values):
        arr = np.array(numbers, dtype=np.float64)
        out: dict[str, object] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        }
        if len(numbers) < len(values):
            out["n"] = len(numbers)
        if keep_values:
            out["values"] = values
        return out
    if all(isinstance(v, list) for v in values) and len({len(v) for v in values}) == 1:
        return [combine([v[i] for v in values], keep_values) for i in range(len(values[0]))]
    return values[0] if all(v == values[0] for v in values) else values


def rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def llm_only(haiku: HaikuSplit) -> dict[str, object]:
    summary = summarize_split(haiku)["new_parser"]
    if not isinstance(summary, dict):
        raise TypeError(f"Haiku summary is {type(summary).__name__}, expected dict")
    return {
        **summary,
        "coverage": 0.0,
        "llm_call_rate": 1.0,
        "llm_cost_usd_per_1k": float(1000 * haiku.cost_usd.mean()),
        "high_conf_oos_misroute_rate": None,
        "note": "no confidence score, so no high-confidence misroute; see oos_misroute_rate",
    }


def analyze_points(
    points: list[Point], results_root: Path, haiku: dict[str, HaikuSplit], log: Log
) -> tuple[dict[str, list], dict[str, list]]:
    """Per-group lists of per-seed (scalars, curves), with the label cross-checks."""
    space = load_label_space()
    reference: dict[str, np.ndarray] = {}
    scalars: dict[str, list] = {}
    curves: dict[str, list] = {}
    test_llm = haiku["test"]
    for point in sorted(points, key=lambda p: (p.group, p.seed)):
        archive = load_logits(results_root / "logits" / point.logits_file)
        for name, split in archive.splits.items():
            ref = reference.setdefault(name, split.labels)
            if not np.array_equal(ref, split.labels):
                raise IncompleteError(f"{point.logits_file}: {name} labels differ from the others")
            check_gold(haiku[name], split.labels, point.logits_file)
        s, c = analyze_run(
            archive.validation, archive.test, point.kind, space, test_llm.pred, test_llm.cost_usd
        )
        scalars.setdefault(point.group, []).append((point, s))
        curves.setdefault(point.group, []).append((point, c))
        log(f"analyzed {point.run_name}")
    return scalars, curves


def group_body(entries: list, keep_values: bool) -> dict[str, object]:
    """One group's seeds merged. Per-seed values are kept for the final router only.

    The ``argmax`` and ``summed`` blocks carry every signal and target for
    both aggregations; with per-seed values they would triple the file,
    so they keep mean and std. The ``final`` block repeats the selected
    aggregation's router numbers with the per-seed values.
    """
    points = [p for p, _ in entries]
    if tuple(p.seed for p in points) != EXPECTED_SEEDS:
        raise IncompleteError(f"{points[0].group}: seeds {[p.seed for p in points]}")
    first = points[0]
    return {
        "model": first.model,
        "k": first.k,
        "oos_train": first.oos_train,
        "kind": first.kind,
        "seeds": list(EXPECTED_SEEDS),
        "run_names": [p.run_name for p in points],
        "logits_sha256": [p.logits_sha256 for p in points],
        "result": {
            key: combine([r[key] for _, r in entries], keep_values and key not in AGGREGATIONS)
            for key in entries[0][1]
        },
    }


def protocol_record() -> dict[str, object]:
    return {
        "aggregations": list(AGGREGATIONS),
        "aggregation_choice": "higher validation 8-way accuracy; ties go to argmax",
        "signals": list(SIGNALS),
        "signals_by_kind": {k: list(v) for k, v in SIGNALS_BY_KIND.items()},
        "target_risks": list(TARGET_RISKS),
        "threshold_choice": (
            "largest validation coverage whose one-sided 95% Wilson upper bound on selective "
            "risk (8-way routing error among kept rows) is <= the target; no such threshold "
            "means every query goes to the LLM"
        ),
        "one_sided_z": ONE_SIDED_95_Z,
        "signal_choice": "largest validation coverage at the target, then lower validation AURC",
        "oracle": "defers exactly the queries the small model gets wrong (8-way)",
        "oos_detection_score": (
            "1 - max in-scope probability at the fitted T (151 intents for argmax, 8 agents "
            "for summed); the four confidence signals are for RQ3"
        ),
        "threshold_result": (
            "main result: thresholds chosen on validation only, reported as they do on test "
            "(option a); diagnostics.sensitivity_reweighted_validation is option (b), "
            "diagnosis only"
        ),
        "tie_at_tau": "a row whose confidence equals tau is kept by the small model",
        "error_recovery_rate": "small-model errors deferred and fixed by Haiku / small errors",
        "recoverable_caught": "errors deferred and fixed by Haiku / errors Haiku would fix",
        "high_conf_oos_misroute_rate": (
            "gold OOS kept by the small model (confidence >= tau) and routed to an in-scope "
            "agent / gold OOS"
        ),
        "coverage_grid": list(COVERAGE_GRID),
        "operating_curve_tau": "the validation score at each coverage of the grid",
        "std": "sample standard deviation (ddof=1) over seeds 42, 43, 44; no other randomness",
        "oos_precision_when_none_predicted": 0.0,
    }


def write_checked(
    path: Path, body: dict, check: Callable[[dict], None], indent: int | None = 1
) -> None:
    """Write via a temporary file that is read back and checked before it replaces ``path``.

    ``indent=None`` writes one line (the curves file: long numeric series).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(rounded(body), indent=indent, separators=(",", ":"), allow_nan=False)
    tmp.write_text(text + "\n", encoding="utf-8")
    try:
        check(json.loads(tmp.read_text(encoding="utf-8")))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def check_groups(expected: set[str]) -> Callable[[dict], None]:
    def check(body: dict) -> None:
        groups = body["groups"]
        if set(groups) != expected:
            raise IncompleteError(f"written groups {sorted(groups)} != expected {sorted(expected)}")
        for name, group in groups.items():
            if group["seeds"] != list(EXPECTED_SEEDS) or len(group["run_names"]) != 3:
                raise IncompleteError(f"{name}: written with seeds {group['seeds']}")

    return check


def check_haiku(body: dict) -> None:
    for split, rows in EXPECTED_ROWS.items():
        if body["splits"][split]["new_parser"]["n"] != rows:
            raise IncompleteError(f"haiku.json: {split} does not hold {rows} rows")


def input_record(results_root: Path, haiku_sha: str) -> dict[str, object]:
    indexes = {s.file: sha256_of(results_root / "curves" / s.file) for s in INDEXES}
    return {
        "archives": EXPECTED_ARCHIVES,
        "logits_manifest_sha256": sha256_of(results_root / MANIFEST_NAME),
        "curve_index_sha256": indexes,
        "haiku_predictions_sha256": haiku_sha,
        "llm_rows": EXPECTED_LLM_ROWS,
    }


ABLATION_PAIR = {"oos_250": "modernbert/k100", "oos_0": "modernbert-oos0/k100"}
ABLATION_AGGREGATION = "argmax"


def mean_std(value: dict | None) -> dict[str, object] | None:
    return None if value is None else {"mean": value["mean"], "std": value["std"]}


def ablation_side(group: dict) -> dict[str, object]:
    result = group["result"]
    fixed = result[ABLATION_AGGREGATION]["oos_detection"]["test"]
    final = result["final"]
    routers = {}
    for target, entry in final["fallback"].items():
        hybrid = entry["hybrid"]["test"]
        routers[target] = {
            k: mean_std(hybrid[k])
            for k in ("oos_recall", "high_conf_oos_misroute_rate", "llm_call_rate", "accuracy_8")
        }
    return {
        "oos_detection_test": {k: mean_std(fixed[k]) for k in ("auroc", "auprc")},
        "small_only_oos_recall": mean_std(final["small_only"]["oos_recall"]),
        "hybrid_test": routers,
    }


def ablation_comparison(groups: dict[str, dict]) -> dict[str, object]:
    """ModernBERT k=100 with 250 vs 0 OOS training rows: same score, same aggregation."""
    return {
        "detection_aggregation": ABLATION_AGGREGATION,
        "detection_score": "1 - max in-scope probability at the fitted T",
        "routers": "final router (validation-selected aggregation and signal)",
        **{side: ablation_side(groups[name]) for side, name in ABLATION_PAIR.items()},
    }


def run(results_root: Path, log: Log = print) -> dict[str, int]:
    points = collect_points(results_root)
    haiku = load_haiku(results_root)
    scalars, curves = analyze_points(points, results_root, haiku, log)
    haiku_sha = sha256_of(full_target(results_root).predictions)
    inputs = input_record(results_root, haiku_sha)
    expected = {p.group for p in points}
    out = results_root / "analysis"
    summary = {
        "format_version": FORMAT_VERSION,
        "inputs": inputs,
        "protocol": protocol_record(),
        "llm_only_test": llm_only(haiku["test"]),
        "groups": {g: group_body(e, keep_values=True) for g, e in sorted(scalars.items())},
    }
    summary["ablation_comparison"] = ablation_comparison(summary["groups"])
    write_checked(out / "summary.json", summary, check_groups(expected))
    curve_body = {
        "format_version": FORMAT_VERSION,
        "inputs": inputs,
        "coverage_grid": list(COVERAGE_GRID),
        "groups": {g: group_body(e, keep_values=False) for g, e in sorted(curves.items())},
    }
    write_checked(out / "curves.json", curve_body, check_groups(expected), indent=None)
    haiku_body = {
        "format_version": FORMAT_VERSION,
        "haiku_predictions_sha256": haiku_sha,
        "splits": {s: summarize_split(h) for s, h in haiku.items()},
    }
    write_checked(out / "haiku.json", haiku_body, check_haiku)
    return {"archives": len(points), "llm_rows": EXPECTED_LLM_ROWS, "groups": len(expected)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log: Log = (lambda _msg: None) if args.quiet else print
    done = run(Path(args.results_root), log)
    print(
        f"completed analysis ({done['archives']}/{EXPECTED_ARCHIVES} archives, "
        f"{done['llm_rows']}/{EXPECTED_LLM_ROWS} llm rows, {done['groups']} groups)"
    )


if __name__ == "__main__":
    main()
