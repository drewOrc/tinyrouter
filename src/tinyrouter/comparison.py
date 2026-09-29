"""AC1b comparison: a rerun's outputs against the committed original ones (docs/PLAN.md 5.1).

Pure functions over already-written JSON; nothing here trains, calls an
API or writes into ``results/``. ``reproduce.py`` calls ``build`` and
writes the result to ``reproduction/<id>/comparison.{json,md}``.

The verdict follows the frozen AC1b criteria and nothing else:

- ``FAIL`` when the flow did not finish (a step failed or never ran, or a
  Haiku check failed) or when AC2 failed (any seed's 150-way in-scope test
  accuracy below 95.7%);
- ``PASS`` otherwise, with the number of ``REVIEW REQUIRED`` items. A
  number outside the original mean ± std needs an explanation, not a
  retuned rerun, and never turns the verdict into FAIL.

How one number is compared (``compare_stat``):

- the original has a spread (a std above 0): the rerun's mean must lie in
  [mean - std, mean + std], otherwise ``REVIEW REQUIRED``;
- the original has no spread (a single run such as Haiku, a std of 0, or a
  std of null because only one seed had a feasible threshold): there is
  nothing to scale a difference by, so any difference is
  ``REVIEW REQUIRED``, and so is a different number of seeds behind it;
- present in only one of the two runs: ``REVIEW REQUIRED``; absent in both: ``OK``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from tinyrouter.ac2 import SEEDS as AC2_SEEDS
from tinyrouter.ac2 import THRESHOLD as AC2_THRESHOLD

OK = "OK"
REVIEW = "REVIEW REQUIRED"
PASS = "PASS"
FAIL = "FAIL"
EXACT_TOLERANCE = 1e-9
EXPECTED_LLM_ROWS = 8600
# docs/PLAN.md 5.1: the original AC6 spend is fixed; the rerun has its own cap.
ORIGINAL_AC6_USD = 3.19
REPRODUCTION_CAP_USD = 5.0
TARGET = "0.02"
CURVE_MODELS = ("modernbert", "bert")
CURVE_KS = (1, 5, 10, 25, 50, 100)
K100 = ("groups", "modernbert/k100", "result")
K10 = ("groups", "modernbert/k10", "result")
DIAG = (*K100, "diagnostics", TARGET)
K10_HYBRID = (*K10, "final", "fallback", TARGET, "hybrid", "test")
K100_HYBRID = (*K100, "final", "fallback", TARGET, "hybrid", "test")
# Every number on the README's first screen (tests/test_report.py FIRST_SCREEN).
HEADLINE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ModernBERT k=100 small-only 8-way accuracy", (*K100, "final", "small_only", "accuracy_8")),
    ("Haiku 4.5 8-way accuracy", ("llm_only_test", "accuracy_8")),
    ("ModernBERT k=10 small-only 8-way accuracy", (*K10, "final", "small_only", "accuracy_8")),
    ("ModernBERT k=10 hybrid 8-way accuracy", (*K10_HYBRID, "accuracy_8")),
    ("ModernBERT k=10 hybrid Haiku call rate", (*K10_HYBRID, "llm_call_rate")),
    ("k=100 threshold, test selective risk", (*DIAG, "test", "selective_risk")),
    ("k=100 threshold, validation selective risk", (*DIAG, "validation", "selective_risk")),
    ("validation OOS share", (*DIAG, "validation_oos_share")),
    ("test OOS share", (*DIAG, "test_oos_share")),
    ("share of the gap explained by OOS share", (*DIAG, "share_of_gap_explained_by_oos_share")),
    ("k=100 kept-OOS error rate, test", (*DIAG, "test", "kept_oos_error_rate")),
    ("k=100 kept-OOS error rate, validation", (*DIAG, "validation", "kept_oos_error_rate")),
    ("ModernBERT k=100 hybrid Haiku call rate", (*K100_HYBRID, "llm_call_rate")),
    ("ModernBERT k=100 hybrid 8-way accuracy", (*K100_HYBRID, "accuracy_8")),
)


def at(tree: object, path: tuple[str, ...]) -> object:
    """The node at ``path``, or None when any key on the way is missing or null."""
    node = tree
    for key in path:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]
    return node


def as_stat(value: object) -> dict | None:
    """A ``{mean, std, n}`` view of a combined statistic or a single number; None if neither."""
    if isinstance(value, dict) and isinstance(value.get("mean"), int | float):
        n = value.get("n", len(value["values"]) if isinstance(value.get("values"), list) else None)
        return {"mean": float(value["mean"]), "std": value.get("std"), "n": n}
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {"mean": float(value), "std": None, "n": 1}
    return None


def compare_stat(label: str, original: object, reproduced: object) -> dict[str, object]:
    """One row of the comparison; see the module docstring for the rule."""
    orig, rep = as_stat(original), as_stat(reproduced)
    row: dict[str, object] = {
        "metric": label,
        "original_mean": None if orig is None else orig["mean"],
        "original_std": None if orig is None else orig["std"],
        "reproduced_mean": None if rep is None else rep["mean"],
        "difference": None if orig is None or rep is None else rep["mean"] - orig["mean"],
    }
    if orig is None or rep is None:
        both_absent = orig is None and rep is None
        rule = "absent in both runs" if both_absent else "present in only one run"
        return {**row, "status": OK if both_absent else REVIEW, "rule": rule}
    difference = abs(rep["mean"] - orig["mean"])
    std = orig["std"]
    if not isinstance(std, int | float) or std <= 0:
        same = difference <= EXACT_TOLERANCE and rep["n"] == orig["n"]
        rule = "original has no seed spread (single run, std 0 or null): any difference is reviewed"
        return {**row, "status": OK if same else REVIEW, "rule": rule}
    inside = difference <= std + EXACT_TOLERANCE
    return {**row, "status": OK if inside else REVIEW, "rule": "within original mean ± std"}


def headline_rows(original: dict, reproduced: dict) -> list[dict]:
    return [compare_stat(label, at(original, p), at(reproduced, p)) for label, p in HEADLINE]


def curve_rows(original: dict, reproduced: dict) -> list[dict]:
    """Test 8-way accuracy and OOS recall of the small model at every k, both encoders."""
    rows = []
    for model in CURVE_MODELS:
        for k in CURVE_KS:
            small = ("groups", f"{model}/k{k}", "result", "final", "small_only")
            for metric in ("accuracy_8", "oos_recall"):
                path = (*small, metric)
                label = f"{model} k={k} small-only {metric}"
                rows.append(compare_stat(label, at(original, path), at(reproduced, path)))
    return rows


def stat_leaves(tree: object, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Paths of every combined statistic (a dict with a ``mean``) under ``tree``."""
    if not isinstance(tree, dict):
        return
    if "mean" in tree:
        yield prefix
        return
    for key, value in tree.items():
        yield from stat_leaves(value, (*prefix, key))


def diagnostic_rows(original: dict, reproduced: dict) -> list[dict]:
    """Every threshold-diagnostic statistic of every encoder point and target risk."""
    rows = []
    groups = sorted(set(original.get("groups", {})) | set(reproduced.get("groups", {})))
    for group in groups:
        base = ("groups", group, "result", "diagnostics")
        paths = sorted(
            set(stat_leaves(at(original, base))) | set(stat_leaves(at(reproduced, base)))
        )
        for path in paths:
            full = (*base, *path)
            label = f"{group} {'/'.join(path)}"
            rows.append(compare_stat(label, at(original, full), at(reproduced, full)))
    return rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def haiku_disagreements(original: list[dict], reproduced: list[dict]) -> dict[str, int]:
    """Rows matched on (split, index): predicted agents that differ, rows in one run only."""
    before = {(r["split"], r["index"]): r["agent"] for r in original}
    after = {(r["split"], r["index"]): r["agent"] for r in reproduced}
    shared = before.keys() & after.keys()
    return {
        "compared_rows": len(shared),
        "different_predictions": sum(before[key] != after[key] for key in shared),
        "only_in_original": len(before.keys() - after.keys()),
        "only_in_reproduction": len(after.keys() - before.keys()),
    }


def haiku_section(
    original_rows: list[dict] | None,
    reproduced_rows: list[dict] | None,
    reproduced_summary: dict | None,
    original_summary: dict,
) -> dict[str, object]:
    """Row count, prediction differences, parser failures and spend of the rerun."""
    if reproduced_rows is None or reproduced_summary is None:
        return {"reached": False, "checks_passed": False, "problems": ["no reproduction run"]}
    totals = reproduced_summary.get("totals", {})
    spent = float(totals.get("cost_usd", 0.0))
    problems = []
    if len(reproduced_rows) != EXPECTED_LLM_ROWS:
        problems.append(f"{len(reproduced_rows)}/{EXPECTED_LLM_ROWS} rows")
    if reproduced_summary.get("identity_sha256") != original_summary.get("identity_sha256"):
        problems.append("identity differs from the original run (model, prompt or settings)")
    if spent > REPRODUCTION_CAP_USD:
        problems.append(f"spent US${spent:.4f}, above the US${REPRODUCTION_CAP_USD:g} cap")
    body: dict[str, object] = {
        "reached": True,
        "rows": f"{len(reproduced_rows)}/{EXPECTED_LLM_ROWS}",
        "parse_failed": sum(bool(r["parse_failed"]) for r in reproduced_rows),
        "original_parse_failed": None,
        "reproduction_cost_usd": round(spent, 6),
        "checks_passed": not problems,
        "problems": problems,
    }
    if original_rows is not None:
        body["original_parse_failed"] = sum(bool(r["parse_failed"]) for r in original_rows)
        body["predictions"] = haiku_disagreements(original_rows, reproduced_rows)
    return body


def budget_section(reproduced_summary: dict | None) -> dict[str, object]:
    """Two separate budgets; the rerun's spend is never added to the original experiment's."""
    spent = None
    if reproduced_summary is not None:
        spent = round(float(reproduced_summary.get("totals", {}).get("cost_usd", 0.0)), 6)
    return {
        "original_ac6_experiment": {
            "usd": ORIGINAL_AC6_USD,
            "note": "fixed: full run US$3.18 plus smoke; not changed by any rerun",
        },
        "reproduction_validation": {
            "usd": spent,
            "cap_usd": REPRODUCTION_CAP_USD,
            "note": "AC1b rerun only, its own journal and cap; not an experiment cost",
        },
        "rule": "separate budgets; never summed or reported as one number",
    }


def ac2_section(reproduced_ac2: dict | None, original_ac2: dict) -> dict[str, object]:
    """Each seed's 150-way in-scope test accuracy against the 95.7% threshold."""
    seeds = {}
    for seed in AC2_SEEDS:
        entry = at(reproduced_ac2, ("seeds", str(seed), "test", "in_scope_accuracy_150"))
        before = at(original_ac2, ("seeds", str(seed), "test", "in_scope_accuracy_150"))
        value = float(entry) if isinstance(entry, int | float) else None
        seeds[str(seed)] = {
            "reproduced": value,
            "original": before,
            "passed": value is not None and value >= AC2_THRESHOLD,
        }
    passed = all(s["passed"] for s in seeds.values())
    return {"threshold": AC2_THRESHOLD, "seeds": seeds, "verdict": PASS if passed else FAIL}


def pilot_section(original: dict[str, dict | None], reproduced: dict[str, dict | None]) -> dict:
    """Selected values; a different choice is reviewed, and configs/curve.yaml is not changed."""
    out = {}
    for kind in ("lr", "steps"):
        before = at(original.get(kind), ("selected",))
        after = at(reproduced.get(kind), ("selected",))
        same = after is not None and after == before
        out[kind] = {
            "original": before,
            "reproduced": after,
            "status": OK if same else REVIEW,
            "note": "the rerun kept the committed configs/curve.yaml either way",
        }
    return out


def flow_passed(steps: list[dict], expected: list[str]) -> bool:
    """Every expected step ran (or was resumed) and passed, in order."""
    passed = [s["name"] for s in steps if s.get("status") == PASS]
    return passed == expected


def verdict(flow_ok: bool, haiku: dict, ac2: dict) -> str:
    if not flow_ok or not haiku.get("checks_passed") or ac2["verdict"] != PASS:
        return FAIL
    return PASS


def review_count(sections: dict[str, object]) -> int:
    """Rows and pilot choices marked REVIEW REQUIRED."""
    count = 0
    for name in ("headline", "learning_curves", "threshold_diagnostics"):
        rows = sections.get(name) or []
        count += sum(row["status"] == REVIEW for row in rows)  # type: ignore[union-attr]
    pilots = sections.get("pilots") or {}
    count += sum(p["status"] == REVIEW for p in pilots.values())  # type: ignore[union-attr]
    return count


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build(
    *,
    original_root: Path,
    reproduced_root: Path,
    original_haiku: Path,
    steps: list[dict],
    expected_steps: list[str],
    context: dict[str, object],
) -> dict[str, object]:
    """The whole comparison for one rerun; reads JSON only."""
    before = read_json(original_root / "analysis" / "summary.json") or {}
    after = read_json(reproduced_root / "analysis" / "summary.json")
    rep_jsonl = reproduced_root / "llm" / "haiku-8way.jsonl"
    rep_summary = read_json(reproduced_root / "llm" / "haiku-8way.json")
    haiku = haiku_section(
        read_jsonl(original_haiku) if original_haiku.is_file() else None,
        read_jsonl(rep_jsonl) if rep_jsonl.is_file() else None,
        rep_summary,
        read_json(original_root / "llm" / "haiku-8way.json") or {},
    )
    ac2 = ac2_section(
        read_json(reproduced_root / "ac2.json"), read_json(original_root / "ac2.json") or {}
    )
    sections: dict[str, object] = {
        "pilots": pilot_section(
            {k: read_json(original_root / "pilots" / f"{k}.json") for k in ("lr", "steps")},
            {k: read_json(reproduced_root / "pilots" / f"{k}.json") for k in ("lr", "steps")},
        ),
        "headline": None if after is None else headline_rows(before, after),
        "learning_curves": None if after is None else curve_rows(before, after),
        "threshold_diagnostics": None if after is None else diagnostic_rows(before, after),
    }
    flow_ok = flow_passed(steps, expected_steps)
    return {
        "format_version": 1,
        "criteria": "docs/PLAN.md section 5.1 (frozen 2026-09-29)",
        "verdict": verdict(flow_ok, haiku, ac2),
        "review_required": review_count(sections),
        "flow": {"passed": flow_ok, "expected_steps": expected_steps, "steps": steps},
        "ac2": ac2,
        "haiku": haiku,
        "budget": budget_section(rep_summary),
        **sections,
        **context,
    }


def fmt(value: object, scale: float = 100.0) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{scale * value:.2f}"


def stat_table(rows: list[dict], only_review: bool = False) -> list[str]:
    lines = [
        "| metric | original mean (%) | original std (%) | rerun (%) | difference (pp) | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if only_review and row["status"] != REVIEW:
            continue
        lines.append(
            f"| {row['metric']} | {fmt(row['original_mean'])} | {fmt(row['original_std'])} | "
            f"{fmt(row['reproduced_mean'])} | {fmt(row['difference'])} | {row['status']} |"
        )
    return lines


def flow_lines(body: dict) -> list[str]:
    lines = ["| step | status | completion line(s) |", "|---|---|---|"]
    for step in body["flow"]["steps"]:
        found = "; ".join(f"`{line}`" for line in step.get("completion_found", [])) or "none"
        status = step["status"] + (" (resumed)" if step.get("resumed") else "")
        lines.append(f"| {step['name']} | {status} | {found} |")
    return lines


def render_markdown(body: dict) -> str:
    """comparison.md: the verdict first, then every section of comparison.json."""
    ac2, haiku, budget = body["ac2"], body["haiku"], body["budget"]
    lines = [
        "# AC1b comparison",
        "",
        f"**Verdict: {body['verdict']}** ({body['review_required']} item(s) {REVIEW}). "
        f"Criteria: {body['criteria']}. FAIL only when the flow or AC2 fails.",
        "",
        "## Flow",
        "",
        *flow_lines(body),
        "",
        f"## AC2 (threshold {100 * ac2['threshold']:.1f}%, every seed): {ac2['verdict']}",
        "",
        "| seed | rerun (%) | original (%) | passed |",
        "|---|---|---|---|",
        *(
            f"| {seed} | {fmt(s['reproduced'])} | {fmt(s['original'])} | {s['passed']} |"
            for seed, s in ac2["seeds"].items()
        ),
        "",
        "## Haiku (reproduction-validation run)",
        "",
        "```json",
        json.dumps(haiku, indent=2),
        "```",
        "",
        "## Budget (separate, never summed)",
        "",
        f"- original AC6 experiment: US${budget['original_ac6_experiment']['usd']:.2f} (fixed)",
        f"- reproduction-validation: US${budget['reproduction_validation']['usd']} "
        f"(cap US${budget['reproduction_validation']['cap_usd']:g})",
        "",
    ]
    lines += pilot_and_number_lines(body)
    return "\n".join(lines) + "\n"


def pilot_and_number_lines(body: dict) -> list[str]:
    lines = ["## Pilots", "", "| pilot | original | rerun | status |", "|---|---|---|---|"]
    for kind, p in body["pilots"].items():
        lines.append(f"| {kind} | {p['original']} | {p['reproduced']} | {p['status']} |")
    lines.append("")
    if body["headline"] is None:
        return [*lines, "Numbers: not reached (the analysis step did not finish).", ""]
    diag = body["threshold_diagnostics"]
    flagged = sum(row["status"] == REVIEW for row in diag)
    return [
        *lines,
        "## README first screen (test)",
        "",
        *stat_table(body["headline"]),
        "",
        "## Learning curves (test, small model alone)",
        "",
        *stat_table(body["learning_curves"]),
        "",
        f"## Threshold diagnostics: {flagged} of {len(diag)} flagged (all rows in the JSON)",
        "",
        *stat_table(diag, only_review=True),
        "",
    ]
