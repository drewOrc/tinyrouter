"""Export a fine-tuned classifier to ONNX and compare ONNX Runtime (CPU) with PyTorch.

Compatibility check for RQ7 (docs/PLAN.md section 4): can this encoder be
exported at all, and do the exported logits match? Tries the
``torch.onnx.export`` exporters in order (dynamo, then the TorchScript
one) and records every attempt, including the error message of a failure.
Scores ``--rows`` validation queries with PyTorch (CPU, float32) and with
ONNX Runtime (CPUExecutionProvider) and reports the maximum absolute logit
difference. No quantization here; that is step 6.

Needs the optional group: ``uv run --group onnx python scripts/export_onnx.py ...``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np

from tinyrouter.data import load_split

EXPORTERS = ("dynamo", "torchscript")


def encode(tokenizer: object, texts: list[str], max_length: int) -> dict[str, np.ndarray]:
    batch = tokenizer(  # type: ignore[operator]
        texts, truncation=True, max_length=max_length, padding=True, return_tensors="np"
    )
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


def export(model: object, inputs: dict[str, np.ndarray], path: Path, exporter: str) -> None:
    import torch

    args = (torch.from_numpy(inputs["input_ids"]), torch.from_numpy(inputs["attention_mask"]))
    names = ["input_ids", "attention_mask"]
    if exporter == "dynamo":
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence")
        shapes = {name: {0: batch, 1: seq} for name in names}
        torch.onnx.export(
            model, args, str(path), input_names=names, output_names=["logits"],
            dynamic_shapes=shapes, dynamo=True,
        )  # fmt: skip
    else:
        axes = {name: {0: "batch", 1: "sequence"} for name in names}
        axes["logits"] = {0: "batch"}
        torch.onnx.export(
            model, args, str(path), input_names=names, output_names=["logits"],
            dynamic_axes=axes, dynamo=False, opset_version=17,
        )  # fmt: skip


def onnx_logits(path: Path, inputs: dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wanted = {i.name for i in session.get_inputs()}
    feed = {k: v.astype(np.int64) for k, v in inputs.items() if k in wanted}
    return np.asarray(session.run(["logits"], feed)[0])


def torch_logits(model: object, inputs: dict[str, np.ndarray]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        return model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).logits.numpy()  # type: ignore[operator]


def max_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def attempt(
    model: object, inputs: dict, reference: np.ndarray, other: dict, path: Path, exporter: str
) -> dict:
    """Export once from ``inputs``; check on them and on ``other`` (another batch and length)."""
    started = time.perf_counter()
    try:
        export(model, inputs, path, exporter)
        logits = onnx_logits(path, inputs)
        other_diff = max_diff(onnx_logits(path, other), torch_logits(model, other))
    except Exception as exc:  # noqa: BLE001 - every failure is the result being recorded
        return {
            "exporter": exporter,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
    weights = sum(p.stat().st_size for p in path.parent.glob("*.data"))
    return {
        "exporter": exporter,
        "ok": True,
        "seconds": round(time.perf_counter() - started, 1),
        "onnx_bytes": path.stat().st_size + weights,
        "max_abs_logit_diff": max_diff(logits, reference),
        "other_shape": list(other["input_ids"].shape),
        "other_shape_max_abs_logit_diff": other_diff,
        "argmax_agree": int((logits.argmax(1) == reference.argmax(1)).sum()),
        "rows": int(reference.shape[0]),
    }


def main(argv: list[str] | None = None) -> None:
    import onnx
    import onnxruntime
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    texts = list(load_split("validation").texts[: args.rows])
    inputs = encode(tokenizer, texts, args.max_length)
    reference = torch_logits(model, inputs)
    longest = max(load_split("validation").texts, key=len)
    other = encode(tokenizer, [longest, texts[0], "hi"], args.max_length)
    attempts = []
    for exporter in EXPORTERS:
        out_dir = model_dir.parent / f"onnx-{exporter}"
        out_dir.mkdir(exist_ok=True)
        attempts.append(attempt(model, inputs, reference, other, out_dir / "model.onnx", exporter))
        print(json.dumps(attempts[-1], indent=2))
    record = {
        "model_type": model.config.model_type,
        "rows": len(texts),
        "padded_sequence_length": int(inputs["input_ids"].shape[1]),
        "attempts": attempts,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "machine": platform.platform(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
