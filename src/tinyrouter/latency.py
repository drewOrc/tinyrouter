"""CPU batch-1 inference latency of the two encoders, and Haiku's measured latency (AC5).

The curve weights were deleted after their logits were archived (disk,
docs/PLAN.md section 6). Latency does not depend on the weight values,
only on the architecture and the input shapes, so each encoder is timed
with the same architecture it was fine-tuned from: the pinned pretrained
checkpoint (``model_name`` at ``model_revision`` from its config), a
151-way classification head (randomly initialised, seed 42), and the same
tokenizer and ``max_length``. Fine-tuning changes values, not shapes. To
show the timed model is the trained one in structure, its parameter count
must equal the one the k=100 run recorded, or the benchmark stops.

Method: CPU, batch 1, ``torch.inference_mode()``, a fixed number of
intra-op threads (recorded), ``WARMUP`` untimed queries, then the first
``N`` validation queries in order. Each query is timed twice from the same
pass: ``end_to_end_ms`` is tokenization plus forward plus argmax (what a
router pays per request) and ``forward_ms`` is the forward pass alone.
Percentiles use numpy's default linear interpolation.

Haiku: ``latency_ms`` in the journal is the client-side wall time of the
last attempt of each ``messages.create`` call. It includes the network
round trip and any API queueing, and the run kept up to 6 calls in flight.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tinyrouter.archive import utc_now
from tinyrouter.config import RunConfig, load_config
from tinyrouter.data import load_split
from tinyrouter.labels import load_label_space

ENCODERS = {"bert": "configs/bert-base.yaml", "modernbert": "configs/modernbert-base.yaml"}
DEFAULT_QUERIES = 500
WARMUP = 50
DEFAULT_THREADS = 4
HEAD_SEED = 42
PERCENTILE_METHOD = "linear"
WEIGHTS_NOTE = (
    "timed with the pinned pretrained backbone and a randomly initialised 151-way head, not the "
    "fine-tuned weights (deleted after their logits were archived); latency depends on the "
    "architecture and input shapes, not on the weight values, and the parameter count equals "
    "the trained k=100 run's"
)
HAIKU_NOTE = (
    "client-side wall time of the last attempt of each messages.create call; includes the "
    "network round trip and API queueing; up to 6 calls were in flight"
)


class ArchitectureMismatchError(RuntimeError):
    """The timed model does not have the parameter count of the trained run."""


def latency_summary(values_ms: Sequence[float]) -> dict[str, float | int]:
    """n, mean, p50, p95, min and max of per-query latencies in milliseconds."""
    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"need a non-empty 1-d list of latencies, got shape {values.shape}")
    p50, p95 = np.percentile(values, [50, 95], method=PERCENTILE_METHOD)
    return {
        "n": int(values.size),
        "mean_ms": round(float(values.mean()), 3),
        "p50_ms": round(float(p50), 3),
        "p95_ms": round(float(p95), 3),
        "min_ms": round(float(values.min()), 3),
        "max_ms": round(float(values.max()), 3),
    }


def haiku_latency(journal: Path) -> dict[str, object]:
    """Latency summary of the stored Haiku calls, per split and over both."""
    by_split: dict[str, list[float]] = {}
    retried = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_split.setdefault(row["split"], []).append(float(row["latency_ms"]))
        retried += int(row["attempts"]) > 1
    every = [v for split in sorted(by_split) for v in by_split[split]]
    return {
        "source": journal.name,
        "measures": HAIKU_NOTE,
        "calls_retried": retried,
        "all": latency_summary(every),
        "splits": {split: latency_summary(v) for split, v in sorted(by_split.items())},
    }


def cpu_name() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def trained_parameter_count(results_root: Path, model: str) -> int:
    """Parameter total recorded by the k=100 seed-42 run of ``model``'s curve."""
    index = json.loads((results_root / "curves" / f"{model}.json").read_text(encoding="utf-8"))
    point = next(p for p in index["points"] if p["k"] == 100 and p["seed"] == 42)
    run = json.loads((results_root / "runs" / f"{point['run_name']}.json").read_text("utf-8"))
    return int(run["training"]["parameters"]["total"])


def load_timed_model(config: RunConfig) -> tuple[object, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = load_label_space()
    torch.manual_seed(HEAD_SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=labels.num_intents,
        id2label=dict(enumerate(labels.intent_names)),
        label2id={name: i for i, name in enumerate(labels.intent_names)},
    )
    return model.to("cpu").eval(), tokenizer


def time_queries(
    model: object, tokenizer: object, texts: Sequence[str], max_length: int
) -> tuple[list[float], list[float]]:
    """(end-to-end ms, forward-only ms) per query, batch 1, in order."""
    import torch

    end_to_end, forward = [], []
    with torch.inference_mode():
        for text in texts:
            started = time.perf_counter_ns()
            batch = tokenizer(  # type: ignore[operator]
                [text], truncation=True, max_length=max_length, return_tensors="pt"
            )
            forward_started = time.perf_counter_ns()
            logits = model(**batch).logits  # type: ignore[operator]
            forward_done = time.perf_counter_ns()
            int(logits.argmax(dim=-1)[0])
            finished = time.perf_counter_ns()
            end_to_end.append((finished - started) / 1e6)
            forward.append((forward_done - forward_started) / 1e6)
    return end_to_end, forward


def benchmark_encoder(
    name: str, results_root: Path, texts: Sequence[str], threads: int
) -> dict[str, object]:
    import torch

    config = load_config(ENCODERS[name])
    model, tokenizer = load_timed_model(config)
    total = sum(p.numel() for p in model.parameters())  # type: ignore[attr-defined]
    trained = trained_parameter_count(results_root, name)
    if total != trained:
        raise ArchitectureMismatchError(
            f"{name}: timed model has {total} parameters, the trained k=100 run {trained}"
        )
    torch.set_num_threads(threads)
    time_queries(model, tokenizer, texts[:WARMUP], config.max_length)
    end_to_end, forward = time_queries(model, tokenizer, texts, config.max_length)
    return {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "num_labels": load_label_space().num_intents,
        "max_length": config.max_length,
        "parameters_total": total,
        "parameters_match_trained_run": True,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),  # type: ignore[attr-defined]
        "end_to_end": latency_summary(end_to_end),
        "forward": latency_summary(forward),
    }


def run_cpu_benchmark(results_root: Path, queries: int, threads: int) -> dict[str, object]:
    import torch
    import transformers

    texts = load_split("validation").texts[:queries]
    torch.set_num_interop_threads(1)
    models = {name: benchmark_encoder(name, results_root, texts, threads) for name in ENCODERS}
    return {
        "format_version": 1,
        "method": {
            "device": "cpu",
            "batch_size": 1,
            "mode": "torch.inference_mode",
            "queries": f"validation rows 0 to {queries - 1}, in order",
            "warmup_queries": WARMUP,
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "percentiles": f"numpy.percentile, method={PERCENTILE_METHOD}",
            "end_to_end": "tokenization + forward + argmax",
            "forward": "forward pass only",
            "weights": WEIGHTS_NOTE,
        },
        "hardware": {
            "cpu": cpu_name(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "measured_at": utc_now(),
        "models": models,
    }


def write_json(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=("cpu", "haiku"))
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--out", default=None, help="output file (default under results/efficiency)"
    )
    args = parser.parse_args(argv)
    root = Path(args.results_root)
    if args.target == "cpu":
        body = run_cpu_benchmark(root, args.queries, args.threads)
        out = root / "efficiency" / "cpu_latency.json"
    else:
        body = {"format_version": 1, **haiku_latency(root / "llm" / "haiku-8way.jsonl")}
        out = root / "efficiency" / "haiku_latency.json"
    out = Path(args.out) if args.out else out
    write_json(out, body)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
