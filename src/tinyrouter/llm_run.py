"""Haiku over validation 3,100 + test 5,500, one stored record per query (PLAN §4, AC6).

The old project kept only aggregate counts, so a cascade could not be
built from it. Here every reply is kept: split, row index, SHA-256 of the
query text (the text itself is not stored; CLINC150 is public and pinned,
and the hash proves which row a record belongs to), gold intent and
agent, the raw reply, the parsed agent, ``parse_failed``, tokens, cost,
latency, attempts and the request id.

Files, for target ``haiku-8way`` under ``results/llm/``:

- ``haiku-8way.<id12>.journal.jsonl``: append-only, one line per finished
  call (and per call that failed for good, so it is on record; those are
  retried next run). ``<id12>`` is the start of ``llm.identity_sha256()``,
  so a different model, prompt, temperature or max_tokens starts a new
  journal and never reuses these replies. Rerunning after an interruption
  calls the API only for rows with no successful record.
- ``haiku-8way.jsonl``: the predictions, written once every row has a
  record, sorted by split and index. Not committed (like the logits
  archives); its SHA-256 is in ``results/llm-manifest.json``.
- ``haiku-8way.json``: summary with identity, pricing, tokens and dollars.

Cost cap: before each call the runner reserves an upper bound on that
call's cost (input tokens counted by the API for the prompt plus one byte
per token of query, output at ``max_tokens``) and does not start a call
that could take cumulative spend for this identity past ``--max-usd``.
Actual cost comes from each response's ``usage``.

Completion: the predictions file, read back from disk, must hold exactly
the expected (split, index) pairs once each, all with this identity, and
its SHA-256 must agree on disk, in the manifest and in the summary. Only
then is ``completed 8600/8600 llm predictions`` printed. The expected
row counts are literals here, not read from ``data.SPLIT_FILES``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from tinyrouter import llm
from tinyrouter.archive import git_state, read_manifest, utc_now, write_manifest
from tinyrouter.data import DATASET_REVISION, Split, SplitName, load_split, sha256_of
from tinyrouter.labels import AGENTS, load_label_space

EXPECTED_ROWS: dict[str, int] = {"validation": 3100, "test": 5500}
# What the smoke run's cost is extrapolated to; equals the sum of EXPECTED_ROWS (tested).
FULL_QUERIES = 8600
SMOKE_ROWS: dict[str, int] = {"validation": 20}
SPLIT_ORDER = ("validation", "test")
DEFAULT_MAX_USD = 5.0
DEFAULT_WORKERS = 6
MAX_WORKERS = 8
MANIFEST_NAME = "llm-manifest.json"
RECORD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "split": str,
    "index": int,
    "query_sha256": str,
    "gold_intent": int,
    "gold_agent": str,
    "raw_text": str,
    "agent": str,
    "parse_failed": bool,
    "input_tokens": int,
    "output_tokens": int,
    "cache_creation_input_tokens": int,
    "cache_read_input_tokens": int,
    "cost_usd": float,
    "latency_ms": int,
    "attempts": int,
    "request_id": (str, type(None)),
    "stop_reason": (str, type(None)),
    "identity_sha256": str,
}

Key = tuple[str, int]
Log = Callable[[str], None]


class JournalError(RuntimeError):
    """A stored record contradicts the data or the identity it claims."""


class IncompleteError(RuntimeError):
    """The predictions file is not exactly the expected rows, once each."""


@dataclass(frozen=True)
class Target:
    name: str
    out_dir: Path
    rows: dict[str, int]
    manifest: Path | None

    @property
    def expected_total(self) -> int:
        return sum(self.rows.values())

    def journal(self, identity_sha: str) -> Path:
        return self.out_dir / f"{self.name}.{identity_sha[:12]}.journal.jsonl"

    @property
    def predictions(self) -> Path:
        return self.out_dir / f"{self.name}.jsonl"

    @property
    def summary(self) -> Path:
        return self.out_dir / f"{self.name}.json"


def full_target(results_root: Path) -> Target:
    return Target("haiku-8way", results_root / "llm", EXPECTED_ROWS, results_root / MANIFEST_NAME)


def smoke_target(results_root: Path) -> Target:
    return Target("haiku-8way-smoke", results_root / "llm-smoke", SMOKE_ROWS, None)


@dataclass(frozen=True)
class Query:
    split: str
    index: int
    text: str
    gold_intent: int

    @property
    def key(self) -> Key:
        return (self.split, self.index)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class Outcome:
    records: dict[Key, dict[str, Any]]
    spent_before: float
    spent_now: float = 0.0
    calls_now: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    stopped: str | None = None


def build_queries(splits: dict[str, Split], rows: dict[str, int]) -> list[Query]:
    """The first ``rows[split]`` rows of each split, in split then index order."""
    queries = []
    for name in SPLIT_ORDER:
        if name not in rows:
            continue
        split = splits[name]
        if len(split) < rows[name]:
            raise IncompleteError(f"{name} has {len(split)} rows, need {rows[name]}")
        for i in range(rows[name]):
            queries.append(Query(name, i, split.texts[i], int(split.intents[i])))
    return queries


def make_record(query: Query, pred: llm.LLMPrediction, identity_sha: str) -> dict[str, Any]:
    space = load_label_space()
    return {
        "split": query.split,
        "index": query.index,
        "query_sha256": query.text_sha256,
        "gold_intent": query.gold_intent,
        "gold_agent": AGENTS[int(space.intent_to_agent_id[query.gold_intent])],
        "raw_text": pred.raw_text,
        "agent": pred.agent,
        "parse_failed": not pred.parsed,
        "input_tokens": pred.input_tokens,
        "output_tokens": pred.output_tokens,
        "cache_creation_input_tokens": pred.cache_creation_input_tokens,
        "cache_read_input_tokens": pred.cache_read_input_tokens,
        "cost_usd": pred.cost_usd,
        "latency_ms": pred.latency_ms,
        "attempts": pred.attempts,
        "request_id": pred.request_id,
        "stop_reason": pred.stop_reason,
        "identity_sha256": identity_sha,
        "created_at": utc_now(),
    }


def check_record_fields(record: dict[str, Any], where: str) -> None:
    for name, expected in RECORD_FIELDS.items():
        if name not in record:
            raise JournalError(f"{where}: record has no {name!r}")
        value = record[name]
        wrong_bool = isinstance(value, bool) and expected is not bool
        if wrong_bool or not isinstance(value, expected):
            raise JournalError(f"{where}: {name}={value!r} is not {expected}")


def read_jsonl(path: Path, allow_torn_tail: bool) -> list[dict[str, Any]]:
    """Parse every line; a crash can leave the last journal line half written, and only it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for n, line in enumerate(lines, 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if allow_torn_tail and n == len(lines):
                break
            raise JournalError(f"{path}:{n}: not JSON ({exc})") from None
    return out


def trim_torn_tail(path: Path) -> None:
    """Cut a half-written last line, so the next append starts on a line of its own."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with path.open("r+b") as fh:
            fh.truncate(data.rfind(b"\n") + 1)


def load_journal(path: Path, queries: dict[Key, Query], identity_sha: str) -> dict[Key, dict]:
    """Successful records of this identity, re-parsed with the current ``parse_agent``."""
    if not path.exists():
        return {}
    done: dict[Key, dict[str, Any]] = {}
    for n, record in enumerate(read_jsonl(path, allow_torn_tail=True), 1):
        if record.get("status") == "failed":
            continue
        where = f"{path}:{n}"
        check_record_fields(record, where)
        key = (record["split"], record["index"])
        query = queries.get(key)
        if record["identity_sha256"] != identity_sha:
            raise JournalError(f"{where}: identity {record['identity_sha256']} != {identity_sha}")
        if query is None or record["query_sha256"] != query.text_sha256:
            raise JournalError(f"{where}: {key} does not match the dataset row it names")
        if record["gold_intent"] != query.gold_intent:
            raise JournalError(f"{where}: {key} gold label differs from the dataset")
        if key in done:
            raise JournalError(f"{where}: {key} recorded twice; was the runner started twice?")
        record["agent"], parsed = llm.parse_agent(record["raw_text"])
        record["parse_failed"] = not parsed
        done[key] = record
    return done


def query_cost_bound(query: Query, base_tokens: int) -> float:
    """Upper bound on one call's cost: every query byte a token, output at max_tokens."""
    return llm.cost_usd(base_tokens + len(query.text.encode("utf-8")), llm.MAX_TOKENS)


def announce_estimate(
    todo: list[Query], base_tokens: int, outcome: Outcome, max_usd: float, log: Log
) -> None:
    bound = sum(query_cost_bound(q, base_tokens) for q in todo)
    in_bound = sum(base_tokens + len(q.text.encode("utf-8")) for q in todo)
    log(
        f"estimate: {len(todo)} calls to make; prompt is {base_tokens} input tokens with a "
        f"1-character query; upper bound {in_bound} input + {len(todo) * llm.MAX_TOKENS} output "
        f"tokens = US${bound:.4f}. Already spent US${outcome.spent_before:.4f} on "
        f"{len(outcome.records)} stored records. Cap US${max_usd:.2f}."
    )
    if outcome.spent_before + bound > max_usd:
        log("warning: the upper bound passes the cap; the run stops at the cap if it gets there")


@dataclass
class Session:
    """State shared by the dispatch loop and the per-call bookkeeping of one run."""

    client: Any
    base_tokens: int
    max_usd: float
    workers: int
    outcome: Outcome
    journal: TextIO
    sleep: Callable[[float], None]
    log: Log
    identity_sha: str

    def spent(self) -> float:
        return self.outcome.spent_before + self.outcome.spent_now


def run(
    queries: list[Query],
    client: Any,
    target: Target,
    *,
    max_usd: float,
    workers: int = DEFAULT_WORKERS,
    sleep: Callable[[float], None] = time.sleep,
    log: Log = print,
) -> Outcome:
    """Call the model for every query without a stored record, within the cost cap."""
    identity_sha = llm.identity_sha256()
    by_key = {q.key: q for q in queries}
    journal = target.journal(identity_sha)
    records = load_journal(journal, by_key, identity_sha)
    outcome = Outcome(records, spent_before=sum(r["cost_usd"] for r in records.values()))
    todo = [q for q in queries if q.key not in records]
    if not todo:
        return outcome
    base = llm.prompt_base_tokens(client)
    announce_estimate(todo, base, outcome, max_usd, log)
    journal.parent.mkdir(parents=True, exist_ok=True)
    trim_torn_tail(journal)
    with journal.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        session = Session(client, base, max_usd, workers, outcome, fh, sleep, log, identity_sha)
        dispatch(session, todo, pool)
    return outcome


def dispatch(session: Session, todo: list[Query], pool: ThreadPoolExecutor) -> None:
    """Keep up to ``workers`` calls in flight, never starting one the cap cannot cover."""
    outcome = session.outcome
    pending = list(reversed(todo))
    inflight: dict[Future, tuple[Query, float]] = {}
    reserved = 0.0
    while pending or inflight:
        while pending and len(inflight) < session.workers and outcome.stopped is None:
            bound = query_cost_bound(pending[-1], session.base_tokens)
            if session.spent() + reserved + bound > session.max_usd:
                outcome.stopped = "cost cap"
                break
            query = pending.pop()
            call = pool.submit(llm.classify, query.text, session.client, session.sleep)
            inflight[call] = (query, bound)
            reserved += bound
        if not inflight:
            break
        finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
        for future in finished:
            query, bound = inflight.pop(future)
            reserved -= bound
            settle(session, future, query)


def settle(session: Session, future: Future, query: Query) -> None:
    """Journal one finished call and update the running totals."""
    outcome = session.outcome
    try:
        pred = future.result()
    except llm.LLMCallError as exc:
        failure = {
            "status": "failed",
            "split": query.split,
            "index": query.index,
            "attempts": exc.attempts,
            "retryable": exc.retryable,
            "error": llm.redact(exc.detail),
            "identity_sha256": session.identity_sha,
            "created_at": utc_now(),
        }
        write_line(session.journal, failure)
        outcome.failures.append(failure)
        session.log(f"FAILED {query.key} after {exc.attempts} attempt(s): {failure['error']}")
        if not exc.retryable and outcome.stopped is None:
            outcome.stopped = "non-retryable API error"
        return
    record = make_record(query, pred, session.identity_sha)
    write_line(session.journal, record)
    outcome.records[query.key] = record
    outcome.spent_now += record["cost_usd"]
    outcome.calls_now += 1


def write_line(fh: TextIO, record: dict[str, Any]) -> None:
    fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    fh.flush()


def expected_keys(rows: dict[str, int]) -> set[Key]:
    return {(split, i) for split, n in rows.items() for i in range(n)}


def check_predictions(path: Path, rows: dict[str, int], identity_sha: str) -> int:
    """``path`` holds exactly the expected rows, once each, all of ``identity_sha``."""
    records = read_jsonl(path, allow_torn_tail=False)
    for n, record in enumerate(records, 1):
        check_record_fields(record, f"{path}:{n}")
        if record["identity_sha256"] != identity_sha:
            raise IncompleteError(f"{path}:{n}: identity {record['identity_sha256']}")
    counts = Counter((r["split"], r["index"]) for r in records)
    expected = expected_keys(rows)
    duplicated = sorted(k for k, c in counts.items() if c > 1)
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected)
    if duplicated or missing or unexpected:
        raise IncompleteError(
            f"{path}: expected {len(expected)} unique rows, got {len(records)}; "
            f"missing {missing[:5]}{'...' if len(missing) > 5 else ''} ({len(missing)}), "
            f"duplicated {duplicated[:5]} ({len(duplicated)}), unexpected {unexpected[:5]}"
        )
    return len(records)


def write_predictions(target: Target, records: dict[Key, dict]) -> Path:
    order = {name: i for i, name in enumerate(SPLIT_ORDER)}
    keys = sorted(records, key=lambda k: (order[k[0]], k[1]))
    tmp = target.predictions.with_name(target.predictions.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        for key in keys:
            write_line(fh, records[key])
    try:
        check_predictions(tmp, target.rows, llm.identity_sha256())
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, target.predictions)
    return target.predictions


def totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    token_fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    out: dict[str, Any] = {f: sum(r[f] for r in records) for f in token_fields}
    out["calls"] = len(records)
    out["cost_usd"] = round(sum(r["cost_usd"] for r in records), 6)
    out["parse_failed"] = {
        s: sum(r["parse_failed"] for r in records if r["split"] == s) for s in SPLIT_ORDER
    }
    return out


def summary_body(target: Target, outcome: Outcome, max_usd: float, sha: str) -> dict[str, Any]:
    commit, dirty = git_state()
    return {
        "name": target.name,
        "identity": llm.identity(),
        "identity_sha256": llm.identity_sha256(),
        "dataset_revision": DATASET_REVISION,
        "rows": target.rows,
        "pricing_usd_per_mtok": llm.PRICE_USD_PER_MTOK,
        "pricing_source": llm.PRICING_SOURCE,
        "predictions_file": target.predictions.name,
        "predictions_sha256": sha,
        "totals": totals(outcome.records.values()),
        "last_invocation": {
            "calls": outcome.calls_now,
            "cost_usd": round(outcome.spent_now, 6),
            "max_usd": max_usd,
            "failed_calls": len(outcome.failures),
        },
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at": utc_now(),
    }


def finalize(target: Target, outcome: Outcome, max_usd: float) -> int:
    """Write predictions, summary and manifest entry, then check that all three agree."""
    path = write_predictions(target, outcome.records)
    sha = sha256_of(path)
    body = summary_body(target, outcome, max_usd, sha)
    target.summary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    if target.manifest is not None:
        files = read_manifest(target.manifest)
        files[path.name] = {
            "sha256": sha,
            "bytes": path.stat().st_size,
            "identity_sha256": body["identity_sha256"],
            "rows": target.rows,
            "git_commit": body["git_commit"],
            "created_at": body["created_at"],
        }
        write_manifest(target.manifest, files)
    return verify(target)


def verify(target: Target) -> int:
    """Row check plus SHA-256 agreement between file, summary and (if any) manifest."""
    identity_sha = llm.identity_sha256()
    count = check_predictions(target.predictions, target.rows, identity_sha)
    on_disk = sha256_of(target.predictions)
    summary = json.loads(target.summary.read_text(encoding="utf-8"))
    listed = on_disk
    if target.manifest is not None:
        listed = read_manifest(target.manifest).get(target.predictions.name, {}).get("sha256")
    if not summary.get("predictions_sha256") == listed == on_disk:
        raise IncompleteError(
            f"{target.predictions.name} SHA-256 differs: summary "
            f"{summary.get('predictions_sha256')}, manifest {listed}, file {on_disk}"
        )
    return count


def report_spend(target: Target, outcome: Outcome, log: Log) -> None:
    t = totals(outcome.records.values())
    log(
        f"this run: {outcome.calls_now} calls, US${outcome.spent_now:.4f}; "
        f"stored: {t['calls']}/{target.expected_total} records, {t['input_tokens']} input + "
        f"{t['output_tokens']} output tokens, US${t['cost_usd']:.4f} total"
    )
    if t["calls"]:
        per_query = t["cost_usd"] / t["calls"]
        full = FULL_QUERIES
        log(
            f"mean US${per_query:.6f} per query; at that rate {full} queries cost "
            f"US${per_query * full:.2f}"
        )


def log_redacted(message: str) -> None:
    print(llm.redact(message), flush=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Claude Haiku over CLINC150 (8-way routing).")
    parser.add_argument("--smoke", action="store_true", help="validation rows 0-19 only")
    parser.add_argument("--verify", action="store_true", help="check stored predictions only")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.max_usd <= 0:
        parser.error("--max-usd must be positive")
    return args


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], Any] = llm.make_client,
    load: Callable[[SplitName], Split] = load_split,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    args = parse_args(argv)
    root = Path(args.results_root)
    target = smoke_target(root) if args.smoke else full_target(root)
    what = "llm smoke predictions" if args.smoke else "llm predictions"
    if args.verify:
        print(f"completed {verify(target)}/{target.expected_total} {what}")
        return
    try:
        client = client_factory()
    except llm.MissingKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    splits = {name: load(name) for name in target.rows}
    queries = build_queries(splits, target.rows)
    outcome = run(
        queries,
        client,
        target,
        max_usd=args.max_usd,
        workers=args.workers,
        sleep=sleep,
        log=log_redacted,
    )
    report_spend(target, outcome, log_redacted)
    missing = target.expected_total - len(outcome.records)
    if missing:
        reason = outcome.stopped or f"{len(outcome.failures)} call(s) failed after retries"
        log_redacted(
            f"incomplete: {missing} rows have no reply ({reason}); rerun to continue, "
            "only those rows are called"
        )
        raise SystemExit(1)
    print(f"completed {finalize(target, outcome, args.max_usd)}/{target.expected_total} {what}")


if __name__ == "__main__":
    main()
