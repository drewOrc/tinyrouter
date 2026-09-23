"""Offline training smoke test: a tiny random BERT built locally, no Hub access."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tinyrouter.calibrate import LeakageError, fit_temperature
from tinyrouter.config import RunConfig
from tinyrouter.data import Split
from tinyrouter.evaluate import predict_logits
from tinyrouter.labels import load_label_space
from tinyrouter.train import pick_device, train

WORDS = ["pay", "my", "bill", "book", "a", "flight", "play", "music", "what", "time"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny-bert")
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    config = BertConfig(
        vocab_size=len(WORDS) + 5,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=load_label_space().num_intents,
    )
    BertForSequenceClassification(config).save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root


def synthetic_split(name: str, n: int = 32) -> Split:
    rng = np.random.default_rng(0)
    texts = tuple(" ".join(rng.choice(WORDS, size=4)) for _ in range(n))
    intents = np.arange(n) % 151
    intents[-1] = load_label_space().oos_intent_id
    return Split(name, texts, intents.astype(np.int64))


def smoke_config(model_dir: Path, tmp_path: Path) -> RunConfig:
    return RunConfig(
        model_name=str(model_dir),
        model_revision="main",
        max_steps=2,
        warmup_ratio=0.0,
        learning_rate=1e-2,
        train_batch_size=8,
        eval_batch_size=16,
        max_length=16,
        device="cpu",
        checkpoint_root=str(tmp_path / "ckpt"),
    )


def classifier_weights(model_dir: Path) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.classifier.weight.detach().numpy().copy()


@pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")
def test_train_updates_weights_and_keeps_only_final(tiny_model_dir, tmp_path):
    config = smoke_config(tiny_model_dir, tmp_path)
    out = tmp_path / "ckpt" / config.run_name
    final_dir = train(config, synthetic_split("train"), out)
    assert (final_dir / "model.safetensors").exists()
    assert list(out.glob("checkpoint-*")) == []
    before, after = classifier_weights(tiny_model_dir), classifier_weights(final_dir)
    assert not np.allclose(before, after), "one optimizer step should change the head"

    val = predict_logits(final_dir, synthetic_split("validation"), config)
    assert val.split == "validation"
    assert val.logits.shape == (32, 151)
    assert fit_temperature(val) > 0

    test = predict_logits(final_dir, synthetic_split("test"), config)
    with pytest.raises(LeakageError):
        fit_temperature(test)


def test_train_refuses_a_non_train_split(tiny_model_dir, tmp_path):
    config = smoke_config(tiny_model_dir, tmp_path)
    with pytest.raises(ValueError, match="only accepts the train split"):
        train(config, synthetic_split("test"), tmp_path / "x")


def test_pick_device_resolves_auto_and_rejects_unavailable(monkeypatch):
    import torch

    assert pick_device("cpu") == "cpu"
    assert pick_device("auto") in {"cuda", "mps", "cpu"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        pick_device("cuda")


def test_smoke_config_file_uses_cpu_and_one_step():
    from tinyrouter.config import load_config

    config = load_config(Path(__file__).parent.parent / "configs" / "smoke.yaml")
    assert (config.device, config.max_steps, config.warmup_ratio) == ("cpu", 1, 0.0)
    assert replace(config, seed=7).run_name.endswith("seed7")


@pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")
def test_train_then_evaluate_archives_logits_and_records_training_cost(
    tiny_model_dir, tmp_path, monkeypatch
):
    import json

    from tinyrouter import evaluate as evaluate_module
    from tinyrouter.archive import check_against_manifest, load_logits
    from tinyrouter.evaluate import RunPaths, evaluate

    config = replace(smoke_config(tiny_model_dir, tmp_path), results_root=str(tmp_path / "res"))
    final_dir = train(config, synthetic_split("train"), tmp_path / "ckpt" / config.run_name)

    summary = json.loads((final_dir / "train_summary.json").read_text())
    assert summary["parameters"]["total"] == summary["parameters"]["trainable"] > 0
    assert summary["global_step"] == 2 and summary["train_wall_seconds"] > 0
    assert summary["oos_train_rows"] == 1 and summary["train_rows"] == 32
    memory = summary["peak_memory"]
    assert memory["device"] == "cpu" and memory["process_peak_rss_bytes"] > 10_000_000
    assert memory["samples"] >= 3

    monkeypatch.setattr(evaluate_module, "eval_split", lambda name, _: synthetic_split(name))
    record = evaluate(config, final_dir)
    paths = RunPaths.of(config)
    check_against_manifest(paths.manifest, paths.logits)
    archive = load_logits(paths.logits)
    assert archive.metadata["seed"] == config.seed
    assert archive.metadata["oos_train_rows"] == 1
    assert archive.test.logits.shape == (32, 151)
    assert record["training"] == summary
    assert record["logits"]["file"] == f"{config.run_name}.npz"
    on_disk = json.loads(paths.results_json.read_text())
    assert on_disk["metrics"]["test"]["raw"]["n"] == 32


def test_evaluate_refuses_a_model_dir_without_a_training_summary(tmp_path):
    from tinyrouter.evaluate import read_training_summary

    with pytest.raises(FileNotFoundError, match="train_summary.json"):
        read_training_summary(tmp_path)
