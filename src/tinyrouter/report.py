"""Build results/summary.md from results/runs/*.json. Minimal for now.

Tables are generated, never typed by hand (docs/PLAN.md AC7). With no
results yet this says so and exits 0; the README section it will feed is
not wired up until the first real run exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = ("accuracy_8", "in_scope_accuracy_150", "oos_recall_151", "ece_151")


def render(records: list[dict[str, object]]) -> str:
    header = "| run | T | " + " | ".join(f"test {c}" for c in COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 2)]
    for record in sorted(records, key=lambda r: str(r["run_name"])):
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        calibrated = metrics["test"]["calibrated"]
        cells = " | ".join(f"{calibrated[c]:.4f}" for c in COLUMNS)
        lines.append(f"| {record['run_name']} | {metrics['temperature']:.3f} | {cells} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    runs_dir = results_dir / "runs"
    paths = sorted(runs_dir.glob("*.json")) if runs_dir.is_dir() else []
    if not paths:
        print(f"no results in {runs_dir}/ yet; run `make train evaluate` first")
        return
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    out = results_dir / "summary.md"
    out.write_text(render(records), encoding="utf-8")
    print(f"wrote {out} ({len(records)} runs)")


if __name__ == "__main__":
    main()
