import json
from dataclasses import replace
from pathlib import Path

import pytest

from run_fakes import FakeRuns, fake_fingerprint
from test_pilots import make_protocol
from tinyrouter.curves import CurveError, index_entry, run_curve
from tinyrouter.evaluate import RunPaths
from tinyrouter.protocol import ProtocolError
from tinyrouter.runs import run_one

pytestmark = pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")


@pytest.fixture(autouse=True)
def fake_samples(monkeypatch):
    """The fake runs record fake_fingerprint; the index compares against the same function."""
    import tinyrouter.curves as curves

    monkeypatch.setattr(curves, "expected_fingerprint", fake_fingerprint)


def quiet(_: str) -> None:
    pass


def ac2_runs(protocol) -> None:
    """The three AC2 runs, made with BERT's own config file (lr 5e-5, no k_shot, no S_min)."""
    fake = FakeRuns()
    for seed in (42, 43, 44):
        config = protocol.base("bert").with_seed(seed)
        run_one(config, fake.train, fake.evaluate, quiet, keep_weights=seed == 42)


def by_point(body) -> dict:
    return {(p["k"], p["seed"]): p for p in body["points"]}


def test_bert_curve_reuses_ac2_at_k100_and_trains_the_other_fifteen(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    ac2_runs(protocol)
    fake = FakeRuns()
    body = run_curve(
        "bert", protocol.curve_configs("bert"), protocol, fake.train, fake.evaluate, quiet
    )
    points = by_point(body)
    assert len(points) == 18 and len(fake.trained) == 15
    assert not any("k100" in name for name in fake.trained)
    for seed in (42, 43, 44):
        full = points[(100, seed)]
        assert full["reused_from"] == f"bert-base-uncased-full-seed{seed}"
        assert full["run_name"] == full["reused_from"]
        assert (full["planned_steps"], full["decided_by"], full["global_step"]) == (
            2385,
            "epochs",
            2385,
        )
    assert points[(1, 42)]["decided_by"] == "min_train_steps"
    assert points[(1, 42)]["planned_steps"] == points[(1, 42)]["global_step"] == 400
    assert points[(1, 42)]["epoch_steps"] == 25
    assert points[(50, 43)]["decided_by"] == "epochs"  # 7,625 rows -> 1,195 steps
    on_disk = json.loads((tmp_path / "res" / "curves" / "bert.json").read_text())
    assert on_disk == json.loads(json.dumps(body))


def test_bert_curve_trains_k100_itself_when_the_pilot_picked_another_lr(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=2e-5, modern_lr=2e-5)
    ac2_runs(protocol)
    fake = FakeRuns()
    body = run_curve(
        "bert", protocol.curve_configs("bert"), protocol, fake.train, fake.evaluate, quiet
    )
    assert len(fake.trained) == 18
    assert all(p["reused_from"] is None for p in body["points"])
    assert by_point(body)[(100, 42)]["run_name"] == "bert-base-uncased-k100-seed42"


def test_an_ac2_seed_whose_archive_changed_is_retrained_not_reused(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    ac2_runs(protocol)
    archive = RunPaths.of(protocol.base("bert").with_seed(43)).logits
    archive.write_bytes(archive.read_bytes() + b"x")
    configs = [c for c in protocol.curve_configs("bert") if c.k_shot == 100]
    fake = FakeRuns()
    body = run_curve("bert", configs, protocol, fake.train, fake.evaluate, quiet)
    assert [p["reused_from"] is not None for p in body["points"]] == [True, False, True]


def test_curve_keeps_no_weights_and_leaves_ac2_seed_42_weights(tmp_path):
    protocol = make_protocol(tmp_path, s_min=200, bert_lr=5e-5, modern_lr=2e-5)
    ac2_runs(protocol)
    fake = FakeRuns()
    run_curve("bert", protocol.curve_configs("bert"), protocol, fake.train, fake.evaluate, quiet)
    kept = sorted(p.parent.parent.name for p in (tmp_path / "ckpt").rglob("model.safetensors"))
    assert kept == ["bert-base-uncased-full-seed42"]


def test_rerunning_a_finished_curve_trains_nothing(tmp_path):
    protocol = make_protocol(tmp_path, s_min=200, bert_lr=5e-5, modern_lr=2e-5)
    configs = protocol.curve_configs("modernbert")
    run_curve("modernbert", configs, protocol, FakeRuns().train, FakeRuns().evaluate, quiet)
    again = FakeRuns()
    run_curve("modernbert", configs, protocol, again.train, again.evaluate, quiet)
    assert again.trained == []


def test_changing_s_min_redoes_every_point_instead_of_mixing_protocols(tmp_path):
    protocol = make_protocol(tmp_path, s_min=200, bert_lr=5e-5, modern_lr=2e-5)
    run_curve(
        "modernbert",
        protocol.curve_configs("modernbert"),
        protocol,
        FakeRuns().train,
        FakeRuns().evaluate,
        quiet,
    )
    moved = replace(protocol, min_train_steps=400)
    again = FakeRuns()
    run_curve(
        "modernbert", moved.curve_configs("modernbert"), moved, again.train, again.evaluate, quiet
    )
    # S_min is part of every run's identity, so every point is redone; the index
    # never mixes points made under two protocols.
    assert len(again.trained) == 18


def test_ablation_runs_modernbert_k100_without_oos_rows_and_never_reuses_ac2(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    ac2_runs(protocol)
    fake = FakeRuns()
    body = run_curve(
        "oos-ablation", protocol.ablation_configs(), protocol, fake.train, fake.evaluate, quiet
    )
    assert len(fake.trained) == 3
    assert {p["oos_train_rows"] for p in body["points"]} == {0}
    assert {p["train_rows"] for p in body["points"]} == {15_000}
    assert body["points"][0]["run_name"] == "ModernBERT-base-k100-oos0-seed42"
    assert (tmp_path / "res" / "curves" / "oos-ablation.json").exists()


def test_curve_refuses_to_start_before_the_pilots_are_copied_in(tmp_path):
    protocol = make_protocol(tmp_path, bert_lr=5e-5)
    with pytest.raises(ProtocolError):
        protocol.curve_configs("bert")


def test_a_point_that_ran_another_step_count_is_refused(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    config = protocol.curve_config("bert", 5, 42)
    record = {
        "run_name": config.run_name,
        "logits": {"file": "x.npz", "sha256": "0" * 64},
        "training": {"train_rows": 763, "oos_train_rows": 13, "global_step": 120},
    }
    with pytest.raises(CurveError, match="ran 120 steps"):
        index_entry(config, record, None)


def cli_fakes(monkeypatch, protocol) -> list[str]:
    import tinyrouter.curves as curves

    calls: list[str] = []
    fake = FakeRuns()
    monkeypatch.setattr(curves, "load_protocol", lambda _: protocol)
    monkeypatch.setattr(curves, "run_baselines", lambda root: calls.append(f"baselines {root}"))

    def curve(name, configs, proto):
        calls.append(f"curve {name}")
        return run_curve(name, configs, proto, fake.train, fake.evaluate, quiet)

    monkeypatch.setattr(curves, "run_curve", curve)
    return calls


def test_the_cli_runs_baselines_before_the_curve(tmp_path, monkeypatch):
    import tinyrouter.curves as curves

    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    calls = cli_fakes(monkeypatch, protocol)
    curves.main(["--model", "modernbert"])
    assert calls == [f"baselines {tmp_path / 'res'}", "curve modernbert"]
    assert (tmp_path / "res" / "curves" / "modernbert.json").exists()


def test_the_cli_stops_before_the_baselines_while_a_pilot_value_is_missing(tmp_path, monkeypatch):
    import tinyrouter.curves as curves

    calls = cli_fakes(monkeypatch, make_protocol(tmp_path, bert_lr=5e-5, modern_lr=2e-5))
    with pytest.raises(ProtocolError, match="pilot-steps"):
        curves.main(["--model", "bert"])
    assert calls == []


def test_the_ablation_cli_skips_the_baselines(tmp_path, monkeypatch):
    import tinyrouter.curves as curves

    calls = cli_fakes(monkeypatch, make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5))
    curves.main(["--ablation"])
    assert calls == ["curve oos-ablation"]
    assert Path(tmp_path / "res" / "curves" / "oos-ablation.json").exists()


def test_index_refuses_a_non_reused_point_whose_archive_changed(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    config = protocol.curve_config("modernbert", 5, 42)
    fake = FakeRuns()
    run_one(config, fake.train, fake.evaluate, quiet, keep_weights=False)
    record = json.loads(RunPaths.of(config).results_json.read_text())
    assert index_entry(config, record, None)["run_name"] == config.run_name
    archive = RunPaths.of(config).logits
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(CurveError, match="archive does not match"):
        index_entry(config, record, None)


def test_curve_writes_no_index_when_an_evaluation_leaves_a_broken_archive(tmp_path):
    from tinyrouter.runs import RunIncompleteError

    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    fake = FakeRuns()

    def truncating(config, model_dir):
        record = fake.evaluate(config, model_dir)
        archive = RunPaths.of(config).logits
        archive.write_bytes(archive.read_bytes()[:100])
        return record

    with pytest.raises(RunIncompleteError):
        run_curve(
            "modernbert",
            protocol.curve_configs("modernbert"),
            protocol,
            fake.train,
            truncating,
            quiet,
        )
    assert not (tmp_path / "res" / "curves" / "modernbert.json").exists()
    assert list((tmp_path / "ckpt").rglob("model.safetensors"))


def test_a_point_trained_on_another_sample_than_curve_sample_now_gives_is_refused(tmp_path):
    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    configs = protocol.curve_configs("modernbert")[:3]
    fake = FakeRuns()
    run_curve("modernbert", configs, protocol, fake.train, fake.evaluate, quiet)

    def drifted(config):
        return fake_fingerprint(config) + "-numpy-upgrade"

    with pytest.raises(CurveError, match="sampler's output changed"):
        run_curve("modernbert", configs, protocol, fake.train, fake.evaluate, quiet, drifted)


def test_reused_ac2_points_get_the_fingerprint_computed_now_and_say_so(tmp_path):
    from tinyrouter.curves import BACKFILLED_SAMPLE

    protocol = make_protocol(tmp_path, s_min=400, bert_lr=5e-5, modern_lr=2e-5)
    ac2_runs(protocol)
    configs = [c for c in protocol.curve_configs("bert") if c.k_shot == 100]
    fake = FakeRuns()
    body = run_curve("bert", configs, protocol, fake.train, fake.evaluate, quiet)
    for point, config in zip(body["points"], configs, strict=True):
        assert point["reused_from"] is not None
        assert point["train_sample_sha256"] == fake_fingerprint(config)
        assert point["train_sample_sha256_source"] == BACKFILLED_SAMPLE


def test_the_real_fingerprint_function_draws_the_curve_sample(monkeypatch):
    import tinyrouter.curves as curves
    import tinyrouter.sampling as sampling
    from test_sampling import FULL
    from tinyrouter.sampling import sample_fingerprint, sample_k_shot

    monkeypatch.setattr(sampling, "load_split", lambda name: FULL)
    curves._fingerprint.cache_clear()
    try:
        assert curves._fingerprint(5, 43, None) == sample_fingerprint(sample_k_shot(FULL, 5, 43))
        assert curves._fingerprint(100, 42, 0) == sample_fingerprint(
            sample_k_shot(FULL, 100, 42, oos_override=0)
        )
    finally:
        curves._fingerprint.cache_clear()
