"""Training cost numbers for the AC5 efficiency table: parameters, wall time, peak memory.

Peak memory is measured differently per device, and the numbers are not
comparable across devices. Each result names what was measured:

- cuda: ``torch.cuda.max_memory_allocated`` after ``reset_peak_memory_stats``.
  The allocator tracks the true peak of tensor memory, including
  activations inside a step.
- mps: torch 2.14 has no peak counter for MPS, so the callback samples
  ``torch.mps.driver_allocated_memory()`` (everything Metal holds for the
  process, cached allocator pools included) and
  ``torch.mps.current_allocated_memory()`` (live tensors only) after each
  backward pass and each optimizer step, and keeps the maxima. The driver
  number is the headline: freed activation blocks stay in the pool, so it
  approximates the high-water mark from above. The tensor number is
  sampled between passes and misses activations that are freed before the
  sample, so it is a lower bound.
- cpu: ``resource.getrusage(RUSAGE_SELF).ru_maxrss``, the process's peak
  resident set size since start, which also counts everything loaded
  before training (Python, torch, the dataset). ru_maxrss is bytes on
  macOS and kilobytes on Linux; ``peak_rss_bytes`` converts.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field

Probe = Callable[[], dict[str, int]]


def count_parameters(model: object) -> dict[str, int]:
    params = list(model.parameters())  # type: ignore[attr-defined]
    return {
        "total": int(sum(p.numel() for p in params)),
        "trainable": int(sum(p.numel() for p in params if p.requires_grad)),
    }


def peak_rss_bytes(ru_maxrss: int, platform: str = sys.platform) -> int:
    return ru_maxrss if platform == "darwin" else ru_maxrss * 1024


def mps_probe() -> dict[str, int]:
    import torch

    return {
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "tensor_allocated_bytes": int(torch.mps.current_allocated_memory()),
    }


def cpu_probe() -> dict[str, int]:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": peak_rss_bytes(rss)}


def cuda_probe() -> dict[str, int]:
    import torch

    return {"max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())}


MEASURES = {
    "mps": "sampled after each backward and optimizer step: torch.mps.driver_allocated_memory "
    "(Metal driver, includes allocator cache; headline) and current_allocated_memory "
    "(live tensors; lower bound)",
    "cuda": "torch.cuda.max_memory_allocated after reset_peak_memory_stats (tensor peak)",
    "cpu": "resource.getrusage ru_maxrss: process peak RSS since start, not training only",
}
PROBES: dict[str, Probe] = {"mps": mps_probe, "cuda": cuda_probe, "cpu": cpu_probe}


@dataclass
class MemorySampler:
    """Keeps the per-key maximum of whatever ``probe`` returns each time ``sample`` runs."""

    probe: Probe
    peaks: dict[str, int] = field(default_factory=dict)
    samples: int = 0

    def sample(self) -> None:
        for key, value in self.probe().items():
            self.peaks[key] = max(self.peaks.get(key, 0), int(value))
        self.samples += 1


def start_measurement(device: str) -> MemorySampler:
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    return MemorySampler(PROBES[device])


def memory_report(device: str, sampler: MemorySampler) -> dict[str, object]:
    measured = {"device": device, "measures": MEASURES[device], "samples": sampler.samples}
    return {**measured, **sampler.peaks}


def sampler_callback(sampler: MemorySampler) -> object:
    """HF Trainer callback that samples after backward and after each optimizer step."""
    from transformers import TrainerCallback

    class SampleMemory(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            sampler.sample()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
            sampler.sample()

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            sampler.sample()

    return SampleMemory()
