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
    assert summary["seed"] == config.seed and config.matches(summary["config"])
    assert len(summary["git_commit"]) in (7, 40) or summary["git_commit"] == "unknown"
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
    env = record["environment"]
    assert env["device"] == "cpu" and env["device_name"]
    assert env["deterministic_algorithms"] in (True, False)
    assert record["logits"]["file"] == f"{config.run_name}.npz"
    on_disk = json.loads(paths.results_json.read_text())
    assert on_disk["metrics"]["test"]["raw"]["n"] == 32


def test_evaluate_refuses_a_model_dir_without_a_training_summary(tmp_path):
    from tinyrouter.evaluate import read_training_summary

    with pytest.raises(FileNotFoundError, match="train_summary.json"):
        read_training_summary(tmp_path)


def test_training_arguments_carry_the_config_seed_for_both_init_and_data_order(tmp_path):
    config = replace(smoke_config(Path("unused"), tmp_path), seed=7)
    args = __import__("tinyrouter.train", fromlist=["x"]).training_arguments(
        config, tmp_path, "cpu"
    )
    assert args.seed == args.data_seed == 7


def test_global_seed_is_set_from_the_config_before_the_model_is_built(tmp_path, monkeypatch):
    import torch

    import tinyrouter.train as train_module

    seen: list[int] = []

    def capture(config, labels):
        seen.append(torch.initial_seed())
        raise RuntimeError("stop after model construction point")

    monkeypatch.setattr(train_module, "load_model_and_tokenizer", capture)
    config = replace(smoke_config(Path("unused"), tmp_path), seed=1234)
    with pytest.raises(RuntimeError, match="stop"):
        train(config, synthetic_split("train"), tmp_path / "x")
    assert seen == [1234]


class StubTokenizer:
    """Encodes 'q<N>' as the single id N, so the model can echo it back."""

    def __call__(self, texts, **_):
        import torch
        from transformers import BatchEncoding

        ids = torch.tensor([[int(t[1:])] for t in texts])
        return BatchEncoding({"input_ids": ids})


class StubModel:
    """Puts all logit mass on the column named by the input id: row i predicts intent N_i."""

    def __init__(self):
        from types import SimpleNamespace

        names = load_label_space().intent_names
        self.config = SimpleNamespace(id2label=dict(enumerate(names)))

    def to(self, _):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids):
        from types import SimpleNamespace

        import torch

        logits = torch.full((input_ids.shape[0], 151), -5.0)
        logits[torch.arange(input_ids.shape[0]), input_ids[:, 0]] = 5.0
        return SimpleNamespace(logits=logits)


def test_predicted_logit_rows_stay_aligned_with_their_labels_across_batches(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_: StubTokenizer())
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", lambda *_: StubModel()
    )
    intents = np.array([150, 3, 42, 7, 0, 99, 42, 11, 5, 120, 64], dtype=np.int64)
    split = Split("test", tuple(f"q{i}" for i in intents), intents)
    config = replace(smoke_config(Path("unused"), tmp_path), eval_batch_size=4)
    out = predict_logits(tmp_path, split, config)
    np.testing.assert_array_equal(out.logits.argmax(axis=1), out.labels)
    np.testing.assert_array_equal(out.labels, intents)


def test_model_whose_id2label_differs_from_intent_names_is_refused(tiny_model_dir, tmp_path):
    from tinyrouter.evaluate import LabelMismatchError, check_model_labels

    names = load_label_space().intent_names
    check_model_labels(dict(enumerate(names)), tmp_path)
    swapped = dict(enumerate(names))
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(LabelMismatchError, match="column 0"):
        check_model_labels(swapped, tmp_path)
    # The fixture model was built without id2label (LABEL_0, LABEL_1, ...).
    config = smoke_config(tiny_model_dir, tmp_path)
    with pytest.raises(LabelMismatchError):
        predict_logits(tiny_model_dir, synthetic_split("test"), config)
