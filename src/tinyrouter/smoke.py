"""End-to-end smoke run: real data, tiny random model, one training step, full scoring path.

Checks that every piece connects (download and checksum, label mapping,
training, logits, logits archive and manifest, temperature fit, metrics,
JSON output). The numbers it
prints are meaningless; the model is random and sees about 150 rows.
Everything is written to a temporary directory and removed afterwards.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from tinyrouter.archive import check_against_manifest, load_logits
from tinyrouter.config import load_config
from tinyrouter.evaluate import RunPaths, evaluate
from tinyrouter.train import prepare_train_split, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args(argv)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="tinyrouter-smoke-") as tmp:
        config = replace(
            load_config(args.config),
            checkpoint_root=str(Path(tmp) / "checkpoints"),
            results_root=str(Path(tmp) / "results"),
        )
        train_split = prepare_train_split(config)
        final_dir = train(config, train_split, Path(config.checkpoint_root) / config.run_name)
        record = evaluate(config, final_dir)
        paths = RunPaths.of(config)
        check_against_manifest(paths.manifest, paths.logits)
        archive = load_logits(paths.logits)
        logits_bytes = paths.logits.stat().st_size
        logits_cells = sum(s.logits.size for s in archive.splits.values())
    metrics = record["metrics"]
    assert isinstance(metrics, dict)
    temperature = float(metrics["temperature"])
    if not (math.isfinite(temperature) and temperature > 0):
        raise SystemExit(f"smoke FAILED: temperature {temperature}")
    for split in ("validation", "test"):
        for variant in ("raw", "calibrated"):
            for key, value in metrics[split][variant].items():
                if key == "n" or key.startswith("nll"):
                    continue
                if not 0.0 <= float(value) <= 1.0:
                    raise SystemExit(f"smoke FAILED: {split}/{variant}/{key}={value}")
    val_n = metrics["validation"]["raw"]["n"]
    test_n = metrics["test"]["raw"]["n"]
    print(
        f"smoke OK: train_rows={len(train_split)} val_rows={val_n} test_rows={test_n} "
        f"T={temperature:.3f} elapsed={time.monotonic() - started:.1f}s"
    )
    print(
        f"smoke logits archive: {logits_bytes} bytes for {logits_cells} float32 logits "
        f"({logits_bytes / logits_cells:.2f} bytes each), sha256 in manifest, load_logits OK; "
        f"peak memory {record['training']['peak_memory']}"
    )


if __name__ == "__main__":
    main()
