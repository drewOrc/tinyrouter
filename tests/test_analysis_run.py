import json

import numpy as np
import pytest

from archive_fakes import fake_metadata, fake_splits
from tinyrouter import analysis_run
from tinyrouter.analysis_run import (
    Point,
    analyze_points,
    collect_points,
    combine,
    load_haiku,
    write_checked,
)
from tinyrouter.archive import MANIFEST_NAME, save_logits, write_manifest
from tinyrouter.calibrate import SplitLogits
from tinyrouter.completeness import IncompleteError
from tinyrouter.data import sha256_of
from tinyrouter.haiku import HaikuMismatchError, HaikuSplit
from tinyrouter.labels import AGENTS, load_label_space
from tinyrouter.llm import identity_sha256
from tinyrouter.llm_run import EXPECTED_ROWS, Outcome, full_target, write_outputs

KS, SEEDS = (1, 5, 10, 25, 50, 100), (42, 43, 44)


def test_combine_gives_mean_sample_std_and_values():
    out = combine([{"a": 1.0, "b": "x"}, {"a": 2.0, "b": "x"}, {"a": 6.0, "b": "x"}])
    assert out["a"]["mean"] == pytest.approx(3.0)
    assert out["a"]["std"] == pytest.approx(np.std([1, 2, 6], ddof=1))
    assert out["a"]["values"] == [1.0, 2.0, 6.0] and out["b"] == "x"


def test_combine_keeps_missing_values_visible_and_merges_lists_elementwise():
    assert combine([None, None, None]) is None
    partial = combine([0.5, None, 0.7])
    assert partial["mean"] == pytest.approx(0.6) and partial["n"] == 2
    assert [p["mean"] for p in combine([[1.0, 2.0], [3.0, 4.0]], keep_values=False)] == [2.0, 3.0]
    assert combine(["argmax", "summed"]) == ["argmax", "summed"]


def fake_index_root(tmp_path):
    """Four indexes, 75 points, each with its own (dummy) archive listed in the manifest."""
    root = tmp_path / "results"
    (root / "logits").mkdir(parents=True)
    manifest = {}

    def entry(run_name, **key):
        path = root / "logits" / f"{run_name}.npz"
        path.write_bytes(run_name.encode())
        manifest[path.name] = {"sha256": sha256_of(path)}
        return {
            **key,
            "run_name": run_name,
            "logits_file": path.name,
            "logits_sha256": sha256_of(path),
        }

    indexes = {
        "bert.json": [
            entry(f"b-k{k}-s{s}", k=k, seed=s, oos_train=None) for k in KS for s in SEEDS
        ],
        "modernbert.json": [
            entry(f"m-k{k}-s{s}", k=k, seed=s, oos_train=None) for k in KS for s in SEEDS
        ],
        "oos-ablation.json": [entry(f"a-s{s}", k=100, seed=s, oos_train=0) for s in SEEDS],
        "baselines.json": [
            entry(f"{b}-k{k}-s{s}", baseline=b, k=k, seed=s)
            for b in ("majority", "tfidf-centroid")
            for k in KS
            for s in SEEDS
        ],
    }
    (root / "curves").mkdir()
    for name, points in indexes.items():
        (root / "curves" / name).write_text(json.dumps({"points": points}), encoding="utf-8")
    write_manifest(root / MANIFEST_NAME, manifest)
    return root


def test_collect_points_accepts_the_complete_set(tmp_path):
    points = collect_points(fake_index_root(tmp_path))
    assert len(points) == 75
    assert len({p.group for p in points}) == 25


def drop_first_point(root, index):
    path = root / "curves" / index
    body = json.loads(path.read_text(encoding="utf-8"))
    body["points"] = body["points"][1:]
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.mark.parametrize(
    "index", ["bert.json", "modernbert.json", "oos-ablation.json", "baselines.json"]
)
def test_one_missing_point_in_any_index_stops_the_analysis(tmp_path, index):
    root = fake_index_root(tmp_path)
    drop_first_point(root, index)
    with pytest.raises(IncompleteError):
        collect_points(root)


def test_a_missing_archive_stops_the_analysis(tmp_path):
    root = fake_index_root(tmp_path)
    (root / "logits" / "a-s43.npz").unlink()
    with pytest.raises(IncompleteError, match="missing"):
        collect_points(root)


def test_the_archive_count_is_a_literal_not_derived_from_the_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_run, "EXPECTED_ARCHIVES", 76)
    with pytest.raises(IncompleteError, match="76"):
        collect_points(fake_index_root(tmp_path))


def haiku_record(split, index, gold_intent):
    space = load_label_space()
    agent = AGENTS[int(space.intent_to_agent_id[gold_intent])]
    return {
        "split": split,
        "index": index,
        "query_sha256": "0" * 64,
        "gold_intent": gold_intent,
        "gold_agent": agent,
        "raw_text": agent,
        "agent": agent,
        "parse_failed": False,
        "input_tokens": 300,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cost_usd": 3.25e-4,
        "latency_ms": 500,
        "attempts": 1,
        "request_id": None,
        "stop_reason": "end_turn",
        "identity_sha256": identity_sha256(),
        "created_at": "2026-09-29T00:00:00+00:00",
    }


def fake_haiku_root(tmp_path):
    root = tmp_path / "results"
    records = {
        (s, i): haiku_record(s, i, i % 151) for s, n in EXPECTED_ROWS.items() for i in range(n)
    }
    write_outputs(full_target(root), Outcome(records=records, spent_before=0.0), 5.0)
    return root


def test_load_haiku_reads_all_8600_rows(tmp_path):
    haiku = load_haiku(fake_haiku_root(tmp_path))
    assert {s: h.pred.size for s, h in haiku.items()} == {"validation": 3100, "test": 5500}


def test_one_missing_haiku_row_stops_the_analysis(tmp_path):
    root = fake_haiku_root(tmp_path)
    path = full_target(root).predictions
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="expected 8600 unique rows"):
        load_haiku(root)


def haiku_for(labels, split):
    n = labels.size
    return HaikuSplit(
        split=split,
        gold_intent=labels.copy(),
        pred=np.zeros(n, dtype=np.int64),
        legacy_pred=np.zeros(n, dtype=np.int64),
        parse_failed=np.zeros(n, dtype=bool),
        legacy_hits=np.ones(n, dtype=np.int64),
        cost_usd=np.full(n, 1e-4),
        raw_text=("finance_agent",) * n,
    )


def archive_root(tmp_path, n_archives=3, relabel=None):
    """``n_archives`` real archives sharing labels (with oos rows); ``relabel`` edits one."""
    root = tmp_path / "results"
    splits = fake_splits(seed=0, n_val=40, n_test=60)
    oos = load_label_space().oos_intent_id
    for s in splits.values():
        s.labels[::5] = oos
    points = []
    for i, seed in enumerate(SEEDS[:n_archives]):
        mine = {
            n: SplitLogits(n, np.roll(s.logits, i, axis=0), s.labels.copy())
            for n, s in splits.items()
        }
        if relabel is not None and i == 1:
            mine[relabel].labels[0] = (mine[relabel].labels[0] + 1) % 151
        name = f"m-k1-s{seed}"
        save_logits(root / "logits" / f"{name}.npz", mine, fake_metadata(run_name=name, seed=seed))
        points.append(Point("m/k1", "m", 1, None, seed, "encoder", name, f"{name}.npz", "x"))
    haiku = {n: haiku_for(s.labels, n) for n, s in splits.items()}
    return root, points, haiku


def test_analyze_points_runs_on_archives_that_agree(tmp_path):
    root, points, haiku = archive_root(tmp_path)
    scalars, curves = analyze_points(points, root, haiku, lambda _m: None)
    assert [p.seed for p, _ in scalars["m/k1"]] == list(SEEDS)
    body = analysis_run.group_body(scalars["m/k1"], keep_values=True)
    assert body["result"]["final"]["small_only"]["accuracy_8"]["values"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_archives_whose_labels_differ_stop_the_analysis(tmp_path, split):
    root, points, haiku = archive_root(tmp_path, relabel=split)
    with pytest.raises(IncompleteError, match=f"{split} labels differ"):
        analyze_points(points, root, haiku, lambda _m: None)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_haiku_gold_that_differs_from_the_archives_stops_the_analysis(tmp_path, split):
    root, points, haiku = archive_root(tmp_path)
    haiku[split].gold_intent[3] = (haiku[split].gold_intent[3] + 1) % 151
    with pytest.raises(HaikuMismatchError, match=split):
        analyze_points(points, root, haiku, lambda _m: None)


def test_a_group_without_all_three_seeds_is_refused(tmp_path):
    root, points, haiku = archive_root(tmp_path, n_archives=2)
    scalars, _ = analyze_points(points, root, haiku, lambda _m: None)
    with pytest.raises(IncompleteError, match="seeds"):
        analysis_run.group_body(scalars["m/k1"], keep_values=True)


def test_an_output_that_fails_its_read_back_check_is_not_written(tmp_path):
    out = tmp_path / "analysis" / "summary.json"

    def refuse(_body):
        raise IncompleteError("wrong groups")

    with pytest.raises(IncompleteError):
        write_checked(out, {"groups": {}}, refuse)
    assert not out.exists() and not list(out.parent.iterdir())


def test_outputs_refuse_nan(tmp_path):
    out = tmp_path / "x.json"
    with pytest.raises(ValueError):
        write_checked(out, {"x": float("nan")}, lambda _b: None)
    assert not out.exists()


def test_combine_reports_no_std_when_fewer_than_two_seeds_have_a_value():
    single = combine([None, 0.023, None])
    assert single["mean"] == pytest.approx(0.023)
    assert single["std"] is None and single["n"] == 1
    assert single["values"] == [None, 0.023, None]


def test_two_indexes_pointing_at_one_archive_stop_the_analysis(tmp_path):
    root = fake_index_root(tmp_path)
    path = root / "curves" / "oos-ablation.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    shared = json.loads((root / "curves" / "modernbert.json").read_text(encoding="utf-8"))
    donor = shared["points"][0]
    for field in ("run_name", "logits_file", "logits_sha256"):
        body["points"][0][field] = donor[field]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(IncompleteError, match="75 points on 74 archives"):
        collect_points(root)


def stat(mean):
    return {"mean": mean, "std": 0.0}


def fake_group(argmax_auroc, summed_auroc):
    def block(auroc):
        return {"oos_detection": {"test": {"auroc": stat(auroc), "auprc": stat(auroc / 2)}}}

    hybrid = {
        k: stat(0.1)
        for k in ("oos_recall", "high_conf_oos_misroute_rate", "llm_call_rate", "accuracy_8")
    }
    return {
        "result": {
            "argmax": block(argmax_auroc),
            "summed": block(summed_auroc),
            "final": {
                "small_only": {"oos_recall": stat(0.6)},
                "fallback": {"0.02": {"hybrid": {"test": hybrid}}},
            },
        }
    }


def test_ablation_detection_uses_one_fixed_aggregation_for_both_models():
    groups = {
        "modernbert/k100": fake_group(0.98, 0.5),
        "modernbert-oos0/k100": fake_group(0.97, 0.9),
    }
    out = analysis_run.ablation_comparison(groups)
    assert out["detection_aggregation"] == "argmax"
    assert out["oos_250"]["oos_detection_test"]["auroc"]["mean"] == 0.98
    assert out["oos_0"]["oos_detection_test"]["auroc"]["mean"] == 0.97
    assert out["oos_0"]["hybrid_test"]["0.02"]["llm_call_rate"]["mean"] == 0.1
