"""The AC1b attempt ledger (``docs/ac1b/attempts.json``) and the three-part budget.

docs/PLAN.md 5.1: the original AC6 experiment cost is fixed at US$3.19;
every AC1b attempt is a reproduction-validation run with its own US$5
cap; their spend is reported per attempt and as a reproduction-validation
total, and never added to, or reported as, the original experiment cost.

The cap is enforced per reproduction id: ``make reproduce`` writes the
Haiku journal under ``reproduction/<id>/results`` and runs it with
``MAX_USD=5``, and ``llm_run`` counts only that journal. A new commit gets
a new id, so a new attempt starts from US$0. An attempt that resumes the
same id under the same identity (attempt 2 resumed attempt 1) shares that
id's journal and therefore its cap.
"""

from __future__ import annotations

import json
from pathlib import Path

LEDGER = Path("docs/ac1b/attempts.json")
REPO = Path(__file__).resolve().parents[2]
ORIGINAL_AC6_USD = 3.19
CAP_USD_PER_ATTEMPT = 5.0
RESULTS = ("PASS", "FAIL", "INFRASTRUCTURE INTERRUPTED")
FIELDS = (
    "attempt",
    "commit",
    "reproduction_id",
    "started_utc",
    "finished_utc",
    "result",
    "reason",
    "haiku_usd",
)


class LedgerError(ValueError):
    """The ledger is malformed or breaks a budget rule."""


def validate(ledger: dict) -> dict:
    """Attempts numbered 1..n, a known result, a reason, spend within the per-attempt cap."""
    if ledger.get("original_ac6_usd") != ORIGINAL_AC6_USD:
        raise LedgerError(f"original_ac6_usd must stay {ORIGINAL_AC6_USD}")
    if ledger.get("cap_usd_per_attempt") != CAP_USD_PER_ATTEMPT:
        raise LedgerError(f"cap_usd_per_attempt must be {CAP_USD_PER_ATTEMPT}")
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise LedgerError("no attempts recorded")
    for number, entry in enumerate(attempts, start=1):
        missing = [f for f in FIELDS if f not in entry]
        if missing:
            raise LedgerError(f"attempt {number}: missing {missing}")
        if entry["attempt"] != number:
            raise LedgerError(f"attempt numbers must run 1..n; found {entry['attempt']}")
        if entry["result"] not in RESULTS:
            raise LedgerError(f"attempt {number}: result {entry['result']!r} not in {RESULTS}")
        if not str(entry["reason"]).strip():
            raise LedgerError(f"attempt {number}: a result needs a reason")
        spent = entry["haiku_usd"]
        if not isinstance(spent, int | float) or not 0 <= spent <= CAP_USD_PER_ATTEMPT:
            raise LedgerError(f"attempt {number}: haiku_usd {spent!r} outside 0..cap")
    return ledger


def load(path: Path = REPO / LEDGER) -> dict:
    return validate(json.loads(path.read_text(encoding="utf-8")))


def total_usd(attempts: list[dict]) -> float:
    return round(sum(float(a["haiku_usd"]) for a in attempts), 6)


def budget(ledger: dict, current: dict | None = None) -> dict[str, object]:
    """Original (fixed), each attempt, and the reproduction-validation total.

    ``current`` is the run being compared (``reproduction_id``, ``usd``)
    when its id has no ledger entry yet; it is listed as pending and
    counted in the total. When the id is already in the ledger, the
    ledger's figures are used and ``current`` adds nothing.
    """
    attempts = [dict(a) for a in ledger["attempts"]]
    recorded_ids = {a["reproduction_id"] for a in attempts}
    if current is not None and current["reproduction_id"] not in recorded_ids:
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "commit": current.get("commit"),
                "reproduction_id": current["reproduction_id"],
                "result": "PENDING",
                "reason": "this comparison; add it to docs/ac1b/attempts.json",
                "haiku_usd": current.get("usd") or 0.0,
            }
        )
    return {
        "original_ac6_experiment": {
            "usd": ORIGINAL_AC6_USD,
            "note": "fixed: full run US$3.18 plus smoke; no AC1b attempt changes it",
        },
        "ac1b_attempts": [
            {k: a.get(k) for k in ("attempt", "reproduction_id", "result", "reason", "haiku_usd")}
            for a in attempts
        ],
        "reproduction_validation_total": {
            "usd": total_usd(attempts),
            "note": "sum over AC1b attempts only; never added to the original experiment cost",
        },
        "cap_usd_per_attempt": CAP_USD_PER_ATTEMPT,
        "rule": (
            "reproduction-validation spend is never reported as original experiment cost and "
            "does not change the AC6 US$5 conclusion"
        ),
    }


def markdown(section: dict) -> list[str]:
    """The three-part budget as markdown lines."""
    lines = [
        f"- Original experiment (AC6, fixed): US${section['original_ac6_experiment']['usd']:.2f}",
        f"- AC1b attempts (reproduction-validation, each capped at "
        f"US${section['cap_usd_per_attempt']:g}):",
        "",
        "| attempt | reproduction id | result | reason | Haiku spend |",
        "|---|---|---|---|---|",
    ]
    for a in section["ac1b_attempts"]:
        lines.append(
            f"| {a['attempt']} | `{a['reproduction_id']}` | {a['result']} | {a['reason']} | "
            f"US${a['haiku_usd']:.6f} |"
        )
    total = section["reproduction_validation_total"]["usd"]
    return [
        *lines,
        "",
        f"- Reproduction-validation total (all AC1b attempts): US${total:.6f}",
        "",
        "Reproduction-validation spend is never reported as original experiment cost and does "
        "not change the AC6 conclusion (Haiku spend at most US$5).",
    ]


BEGIN = "<!-- BEGIN AC1b budget, generated from docs/ac1b/attempts.json; do not edit by hand -->"
END = "<!-- END AC1b budget -->"
BUDGET_DOCS = (Path("README.md"), Path("docs/ac1b/README.md"))


def block(ledger: dict) -> str:
    return "\n".join([BEGIN, *markdown(budget(ledger)), END])


def with_block(text: str, ledger: dict) -> str:
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < start:
        raise LedgerError("no AC1b budget block (BEGIN ... END markers)")
    return text[:start] + block(ledger) + text[end + len(END) :]


def main(argv: list[str] | None = None) -> None:
    """Rewrite the budget block in README.md and docs/ac1b/README.md; ``--check`` only compares."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    ledger = load(REPO / LEDGER)
    stale = []
    for doc in BUDGET_DOCS:
        path = REPO / doc
        text = path.read_text(encoding="utf-8")
        new = with_block(text, ledger)
        if new == text:
            continue
        if args.check:
            stale.append(str(doc))
        else:
            path.write_text(new, encoding="utf-8")
            print(f"wrote {doc}")
    if stale:
        print(f"stale, run `python -m tinyrouter.ac1b_ledger`: {', '.join(stale)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
