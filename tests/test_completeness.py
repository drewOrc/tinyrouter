"""Completion checks: a curve, the ablation or the baselines say "completed" only when they are.

Each index is checked twice, before running (the plan) and after (the
index read back from disk plus every archive's SHA-256), and each check
has its own missing-point and duplicated-point test here.
"""

import json
import shutil
from pathlib import Path

import pytest

import tinyrouter.curves as curves
from run_fakes import FakeRuns, fake_fingerprint
from test_baselines import EVALS, TRAIN
from test_pilots import make_protocol
from tinyrouter import baselines, completeness
from tinyrouter.archive import read_manifest, write_manifest
from tinyrouter.completeness import IncompleteError
from tinyrouter.curves import CurveError, run_curve
from tinyrouter.evaluate import RunPaths
from tinyrouter.protocol import SEEDS
from tinyrouter.sampling import CURVE_KS

pytestmark = pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")


@pytest.fixture(autouse=True)
def fake_samples(monkeypatch):
    monkeypatch.setattr(curves, "expected_fingerprint", fake_fingerprint)


@pytest.fixture
def protocol(tmp_path):
    return make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)


class Log:
    def __init__(self, index: Path | None = None) -> None:
        self.lines: list[str] = []
        self.index = index
        self.index_existed_at_completion: bool | None = None

    def __call__(self, line: str) -> None:
        if line.startswith("completed") and self.index is not None:
            self.index_existed_at_completion = self.index.exists()
        self.lines.append(line)

    def completed(self) -> list[str]:
        return [line for line in self.lines if line.startswith("completed")]


def curve_index(tmp_path: Path, name: str) -> Path:
    return tmp_path / "res" / "curves" / f"{name}.json"


def assert_nothing_published(tmp_path: Path, name: str, log: Log) -> None:
    index = curve_index(tmp_path, name)
    assert not index.exists()
    assert not index.with_name(index.name + ".tmp").exists()
    assert log.completed() == []


def test_the_pinned_expectations_match_the_constants_the_loops_use_today():
    assert completeness.EXPECTED_KS == CURVE_KS
    assert completeness.EXPECTED_SEEDS == SEEDS
    assert completeness.EXPECTED_BASELINES == baselines.BASELINES
    assert len(completeness.CURVE_POINTS) == 18
    assert len(completeness.ABLATION_POINTS) == 3
    assert len(completeness.BASELINE_POINTS) == 36


# Encoder curves -------------------------------------------------------------


def test_a_full_curve_says_completed_18_of_18_after_its_index_is_in_place(tmp_path, protocol):
    fake = FakeRuns()
    log = Log(curve_index(tmp_path, "modernbert"))
    run_curve(
        "modernbert", protocol.curve_configs("modernbert"), protocol, fake.train, fake.evaluate, log
    )
    assert log.completed() == ["completed 18/18 encoder points (modernbert)"]
    assert log.lines[-1] == "completed 18/18 encoder points (modernbert)"
    assert log.index_existed_at_completion is True


def test_curve_plan_missing_a_point_is_refused_before_anything_trains(tmp_path, protocol):
    fake, log = FakeRuns(), Log()
    configs = protocol.curve_configs("bert")[:-1]
    with pytest.raises(IncompleteError, match=r"bert plan: .*missing \[\(100, 44, None\)\]"):
        run_curve("bert", configs, protocol, fake.train, fake.evaluate, log)
    assert fake.trained == []
    assert_nothing_published(tmp_path, "bert", log)


def test_curve_plan_repeating_a_point_is_refused_before_anything_trains(tmp_path, protocol):
    fake, log = FakeRuns(), Log()
    configs = protocol.curve_configs("bert")
    configs = [*configs[:-1], configs[0]]  # still 18 configs, 17 distinct
    with pytest.raises(IncompleteError, match=r"duplicated \[\(1, 42, None\)\]"):
        run_curve("bert", configs, protocol, fake.train, fake.evaluate, log)
    assert fake.trained == []
    assert_nothing_published(tmp_path, "bert", log)


def test_curve_plan_for_another_model_is_refused(tmp_path, protocol):
    fake, log = FakeRuns(), Log()
    with pytest.raises(CurveError, match="holds only"):
        run_curve(
            "modernbert", protocol.curve_configs("bert"), protocol, fake.train, fake.evaluate, log
        )
    assert fake.trained == []


def test_an_unknown_index_name_is_refused(protocol):
    with pytest.raises(CurveError, match="unknown curve"):
        run_curve("bert-v2", protocol.curve_configs("bert"), protocol, log=Log())


def tamper_point(monkeypatch, target: tuple[int, int], change) -> None:
    """Let run_point run for real, then pass ``change(entry, config)`` over one point's entry."""
    real = curves.run_point

    def tampered(config, *args):
        entry = real(config, *args)
        if (config.k_shot, config.seed) == target:
            change(entry, config)
        return entry

    monkeypatch.setattr(curves, "run_point", tampered)


def run_modernbert(protocol, log: Log) -> None:
    fake = FakeRuns()
    run_curve(
        "modernbert", protocol.curve_configs("modernbert"), protocol, fake.train, fake.evaluate, log
    )


def test_curve_index_missing_a_point_is_refused_after_running(tmp_path, protocol, monkeypatch):
    tamper_point(monkeypatch, (5, 44), lambda entry, _: entry.update(k=7))
    log = Log()
    with pytest.raises(IncompleteError, match=r"modernbert: .*missing \[\(5, 44, None\)\]"):
        run_modernbert(protocol, log)
    assert_nothing_published(tmp_path, "modernbert", log)


def test_curve_index_repeating_a_point_is_refused_after_running(tmp_path, protocol, monkeypatch):
    tamper_point(monkeypatch, (5, 44), lambda entry, _: entry.update(seed=43))
    log = Log()
    with pytest.raises(IncompleteError, match=r"modernbert: .*duplicated \[\(5, 43, None\)\]"):
        run_modernbert(protocol, log)
    assert_nothing_published(tmp_path, "modernbert", log)


def test_curve_index_is_refused_when_an_archive_changed_after_its_point_ran(
    tmp_path, protocol, monkeypatch
):
    def corrupt(_, config):
        archive = RunPaths.of(config).logits
        archive.write_bytes(archive.read_bytes() + b"x")

    tamper_point(monkeypatch, (10, 43), corrupt)
    log = Log()
    with pytest.raises(IncompleteError, match="SHA-256 differs"):
        run_modernbert(protocol, log)
    assert_nothing_published(tmp_path, "modernbert", log)


def test_curve_index_is_refused_when_the_manifest_disagrees(tmp_path, protocol, monkeypatch):
    def relist(_, config):
        paths = RunPaths.of(config)
        files = read_manifest(paths.manifest)
        files[paths.logits.name]["sha256"] = "0" * 64
        write_manifest(paths.manifest, files)

    tamper_point(monkeypatch, (10, 43), relist)
    log = Log()
    with pytest.raises(IncompleteError, match="SHA-256 differs"):
        run_modernbert(protocol, log)
    assert_nothing_published(tmp_path, "modernbert", log)


def test_curve_index_is_refused_when_an_archive_is_gone(tmp_path, protocol, monkeypatch):
    tamper_point(monkeypatch, (1, 42), lambda _, config: RunPaths.of(config).logits.unlink())
    log = Log()
    with pytest.raises(IncompleteError, match="is missing"):
        run_modernbert(protocol, log)
    assert_nothing_published(tmp_path, "modernbert", log)


def test_a_failed_rerun_leaves_the_previous_verified_index_untouched(
    tmp_path, protocol, monkeypatch
):
    run_modernbert(protocol, Log())
    before = curve_index(tmp_path, "modernbert").read_bytes()
    tamper_point(monkeypatch, (5, 44), lambda entry, _: entry.update(seed=43))
    with pytest.raises(IncompleteError):
        run_modernbert(protocol, Log())
    assert curve_index(tmp_path, "modernbert").read_bytes() == before


# OOS ablation ---------------------------------------------------------------


def test_the_ablation_says_completed_3_of_3(tmp_path, protocol):
    fake = FakeRuns()
    log = Log(curve_index(tmp_path, "oos-ablation"))
    run_curve("oos-ablation", protocol.ablation_configs(), protocol, fake.train, fake.evaluate, log)
    assert log.completed() == ["completed 3/3 ablation points"]
    assert log.index_existed_at_completion is True


def test_ablation_plan_missing_a_seed_is_refused_before_anything_trains(tmp_path, protocol):
    fake, log = FakeRuns(), Log()
    with pytest.raises(IncompleteError, match=r"oos-ablation plan: .*missing \[\(100, 44, 0\)\]"):
        run_curve(
            "oos-ablation",
            protocol.ablation_configs()[:2],
            protocol,
            fake.train,
            fake.evaluate,
            log,
        )
    assert fake.trained == []
    assert_nothing_published(tmp_path, "oos-ablation", log)


def test_ablation_plan_repeating_a_seed_is_refused_before_anything_trains(tmp_path, protocol):
    fake, log = FakeRuns(), Log()
    first, second, _ = protocol.ablation_configs()
    with pytest.raises(IncompleteError, match=r"duplicated \[\(100, 42, 0\)\]"):
        run_curve("oos-ablation", [first, second, first], protocol, fake.train, fake.evaluate, log)
    assert fake.trained == []
    assert_nothing_published(tmp_path, "oos-ablation", log)


def test_ablation_plan_with_oos_rows_is_refused(tmp_path, protocol):
    fake, log = FakeRuns(), Log()
    with_oos = [protocol.curve_config("modernbert", 100, seed) for seed in (42, 43, 44)]
    with pytest.raises(IncompleteError, match=r"unexpected \[\(100, 42, None\)"):
        run_curve("oos-ablation", with_oos, protocol, fake.train, fake.evaluate, log)
    assert fake.trained == []


def run_ablation(protocol, log: Log) -> None:
    fake = FakeRuns()
    run_curve("oos-ablation", protocol.ablation_configs(), protocol, fake.train, fake.evaluate, log)


def test_ablation_index_missing_a_point_is_refused_after_running(tmp_path, protocol, monkeypatch):
    tamper_point(monkeypatch, (100, 44), lambda entry, _: entry.update(oos_train=None))
    log = Log()
    with pytest.raises(IncompleteError, match=r"oos-ablation: .*missing \[\(100, 44, 0\)\]"):
        run_ablation(protocol, log)
    assert_nothing_published(tmp_path, "oos-ablation", log)


def test_ablation_index_repeating_a_point_is_refused_after_running(tmp_path, protocol, monkeypatch):
    tamper_point(monkeypatch, (100, 44), lambda entry, _: entry.update(seed=42))
    log = Log()
    with pytest.raises(IncompleteError, match=r"oos-ablation: .*duplicated \[\(100, 42, 0\)\]"):
        run_ablation(protocol, log)
    assert_nothing_published(tmp_path, "oos-ablation", log)


# Baselines ------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_results(tmp_path_factory):
    """One full baseline run on the synthetic data, shared read-only by the tests below."""
    import tinyrouter.sampling as sampling

    root = tmp_path_factory.mktemp("baselines")
    patch = pytest.MonkeyPatch()
    patch.setattr(sampling, "load_split", lambda name: TRAIN)
    log = Log(root / "curves" / "baselines.json")
    try:
        baselines.run_all(root, eval_fn=EVALS.__getitem__, log=log)
    finally:
        patch.undo()
    return root, log


@pytest.fixture
def baseline_copy(baseline_results, tmp_path, monkeypatch):
    """A private copy of the finished run; rerunning on it refits nothing (every run is intact)."""
    import tinyrouter.sampling as sampling

    monkeypatch.setattr(sampling, "load_split", lambda name: TRAIN)
    root = tmp_path / "results"
    shutil.copytree(baseline_results[0], root)
    return root


def rerun_baselines(root: Path, log: Log) -> None:
    baselines.run_all(root, eval_fn=EVALS.__getitem__, log=log)


def never_load(name: str):
    raise AssertionError(f"loaded {name}: the plan check must come before any work")


def test_the_baselines_say_completed_36_of_36_after_their_index_is_in_place(baseline_results):
    root, log = baseline_results
    assert log.completed() == ["completed 36/36 baseline points"]
    assert log.lines[-1] == "completed 36/36 baseline points"
    assert log.index_existed_at_completion is True
    assert len(json.loads((root / "curves" / "baselines.json").read_text())["points"]) == 36


def test_baseline_plan_missing_a_k_is_refused_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "CURVE_KS", (1, 5, 10, 25, 50))
    log = Log()
    with pytest.raises(IncompleteError, match=r"baselines plan: expected 36 unique points, got 30"):
        baselines.run_all(tmp_path, eval_fn=never_load, log=log)
    assert list(tmp_path.iterdir()) == [] and log.completed() == []


def test_baseline_plan_repeating_a_k_is_refused_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "CURVE_KS", (1, 5, 10, 25, 50, 100, 100))
    log = Log()
    with pytest.raises(IncompleteError, match=r"baselines plan: .*missing \[\], duplicated \[\("):
        baselines.run_all(tmp_path, eval_fn=never_load, log=log)
    assert list(tmp_path.iterdir()) == [] and log.completed() == []


def tamper_baseline(monkeypatch, target: tuple[str, int, int], **changes) -> None:
    real = baselines.index_entry

    def tampered(name, k, seed, record):
        entry = real(name, k, seed, record)
        if (name, k, seed) == target:
            entry.update(changes)
        return entry

    monkeypatch.setattr(baselines, "index_entry", tampered)


def test_baseline_index_missing_a_point_is_refused_after_running(baseline_copy, monkeypatch):
    before = (baseline_copy / "curves" / "baselines.json").read_bytes()
    tamper_baseline(monkeypatch, ("majority", 25, 44), baseline="knn")
    log = Log()
    with pytest.raises(IncompleteError, match=r"baselines: .*missing \[\('majority', 25, 44\)\]"):
        rerun_baselines(baseline_copy, log)
    assert log.completed() == []
    assert (baseline_copy / "curves" / "baselines.json").read_bytes() == before


def test_baseline_index_repeating_a_point_is_refused_after_running(baseline_copy, monkeypatch):
    tamper_baseline(monkeypatch, ("majority", 25, 44), seed=43)
    log = Log()
    with pytest.raises(IncompleteError, match=r"duplicated \[\('majority', 25, 43\)\]"):
        rerun_baselines(baseline_copy, log)
    assert log.completed() == []
    assert not (baseline_copy / "curves" / "baselines.json.tmp").exists()


def test_baseline_index_is_refused_when_an_archive_changes_during_the_run(
    baseline_copy, monkeypatch
):
    real = baselines.index_entry

    def corrupting(name, k, seed, record):
        entry = real(name, k, seed, record)
        if (name, k, seed) == ("tfidf-centroid", 1, 42):
            archive = RunPaths.named(baseline_copy, entry["run_name"]).logits
            archive.write_bytes(archive.read_bytes() + b"x")
        return entry

    monkeypatch.setattr(baselines, "index_entry", corrupting)
    log = Log()
    with pytest.raises(IncompleteError, match="SHA-256 differs"):
        rerun_baselines(baseline_copy, log)
    assert log.completed() == []


# The index checker on its own ------------------------------------------------


def written_index(root: Path, points: list[dict]) -> Path:
    path = root / "check.json"
    path.write_text(json.dumps({"points": points}))
    return path


def test_verify_index_counts_a_complete_baseline_index(baseline_results, tmp_path):
    root, _ = baseline_results
    points = json.loads((root / "curves" / "baselines.json").read_text())["points"]
    path = written_index(tmp_path, points)
    count = completeness.verify_index(
        path, completeness.BASELINE_KEY, completeness.BASELINE_POINTS, root, "baselines"
    )
    assert count == 36


def test_verify_index_refuses_an_index_one_point_short(baseline_results, tmp_path):
    root, _ = baseline_results
    points = json.loads((root / "curves" / "baselines.json").read_text())["points"][:-1]
    with pytest.raises(IncompleteError, match="got 35 .35 unique.; missing"):
        completeness.verify_index(
            written_index(tmp_path, points),
            completeness.BASELINE_KEY,
            completeness.BASELINE_POINTS,
            root,
            "baselines",
        )


def test_verify_index_refuses_an_index_with_a_point_twice(baseline_results, tmp_path):
    root, _ = baseline_results
    points = json.loads((root / "curves" / "baselines.json").read_text())["points"]
    with pytest.raises(IncompleteError, match="got 37 .36 unique.; missing \\[\\], duplicated"):
        completeness.verify_index(
            written_index(tmp_path, [*points, points[0]]),
            completeness.BASELINE_KEY,
            completeness.BASELINE_POINTS,
            root,
            "baselines",
        )
