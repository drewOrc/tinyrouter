"""Moved output roots: a rerun (``make reproduce``) writes beside ``results/``, never into it.

The CLIs of AC2, the pilots and the curves run here with fake training and
evaluation in a scratch directory that holds a copy of ``configs/`` and a
``results/`` with one file in it. After all three, that ``results/`` must be
byte for byte what it was, no ``checkpoints/`` may exist, and every new file
must be under the moved roots.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from run_fakes import FakeRuns, fake_fingerprint
from tinyrouter import ac2, curves, pilots, report
from tinyrouter.ac2 import SetupError
from tinyrouter.config import load_config
from tinyrouter.protocol import load_protocol

pytestmark = pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")

ROOT = Path(__file__).parent.parent
MOVED = ("--results-root", "repro/results", "--checkpoint-root", "repro/ckpt")


def tree(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def quiet(_: str) -> None:
    pass


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ac2.json").write_text('{"committed": true}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curves, "expected_fingerprint", fake_fingerprint)
    return tmp_path


def test_load_config_moves_only_the_two_location_fields():
    plain = load_config(ROOT / "configs" / "bert-base.yaml")
    moved = load_config(ROOT / "configs" / "bert-base.yaml", results_root="r", checkpoint_root="c")
    assert (moved.results_root, moved.checkpoint_root) == ("r", "c")
    assert moved.identity() == plain.identity()
    assert moved.run_name == plain.run_name


def test_load_protocol_moves_every_base_config_and_so_every_curve_and_pilot_run():
    protocol = load_protocol(ROOT / "configs" / "curve.yaml", results_root="r", checkpoint_root="c")
    configs = [*protocol.curve_configs("bert"), *protocol.ablation_configs()]
    configs += [p.config for p in pilots.lr_points(protocol) + pilots.steps_points(protocol)]
    assert {c.results_root for c in configs} == {"r"}
    assert {Path(c.checkpoint_root).parts[0] for c in configs} == {"c"}
    assert pilots.pilot_output(protocol, "lr") == Path("r/pilots/lr.json")


def test_ac2_pilots_and_curves_with_moved_roots_leave_results_untouched(scratch, monkeypatch):
    before = tree(scratch / "results")
    fake = FakeRuns()
    monkeypatch.setattr(ac2, "default_train", fake.train)
    monkeypatch.setattr(ac2, "default_evaluate", fake.evaluate)
    with pytest.raises(SetupError, match="test rows"):  # fake splits are not 5,500 rows
        ac2.main(["--config", "configs/bert-base.yaml", *MOVED])
    real_pilot = pilots.run_pilot
    monkeypatch.setattr(
        pilots,
        "run_pilot",
        lambda kind, protocol, out: real_pilot(
            kind, protocol, out, fake.train, fake.validation, quiet
        ),
    )
    pilots.main(["lr", *MOVED])
    baseline_roots: list[Path] = []
    monkeypatch.setattr(curves, "run_baselines", baseline_roots.append)
    real_curve = curves.run_curve
    monkeypatch.setattr(
        curves,
        "run_curve",
        lambda name, configs, proto: real_curve(
            name, configs, proto, fake.train, fake.evaluate, quiet
        ),
    )
    curves.main(["--model", "bert", *MOVED])

    assert tree(scratch / "results") == before
    assert not (scratch / "checkpoints").exists()
    assert baseline_roots == [Path("repro/results")]
    outside = [p for p in tree(scratch) if not p.startswith(("repro/", "configs/", "results/"))]
    assert outside == []
    moved = scratch / "repro" / "results"
    assert len(list((moved / "runs").glob("bert-base-uncased-full-seed*.json"))) == 3
    lr = json.loads((moved / "pilots" / "lr.json").read_text())
    reused = [p for p in lr["points"] if p["source"].startswith("reused")]
    assert [p["run_name"] for p in reused] == ["bert-base-uncased-k100-seed42"]
    index = json.loads((moved / "curves" / "bert.json").read_text())
    k100 = [p for p in index["points"] if p["k"] == 100]
    assert {p["reused_from"] for p in k100} == {
        f"bert-base-uncased-full-seed{s}" for s in (42, 43, 44)
    }


@pytest.fixture
def report_copy(tmp_path):
    shutil.copytree(
        ROOT / "results", tmp_path / "moved", ignore=shutil.ignore_patterns("logits", "*.jsonl")
    )
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    return tmp_path


def test_report_with_readme_out_leaves_the_repo_readme_alone(report_copy):
    before = (report_copy / "README.md").read_bytes()
    out = report_copy / "rerun" / "README.md"
    report.main(
        ["--results-dir", str(report_copy / "moved"), "--repo", str(report_copy)]
        + ["--readme-out", str(out)]
    )
    assert (report_copy / "README.md").read_bytes() == before
    assert out.read_bytes() == before  # same JSON gives the same README


def test_report_refuses_moved_results_without_readme_out(report_copy):
    before = (report_copy / "README.md").read_bytes()
    with pytest.raises(SystemExit) as exc:
        report.main(["--results-dir", str(report_copy / "moved"), "--repo", str(report_copy)])
    assert exc.value.code == 2
    assert (report_copy / "README.md").read_bytes() == before
