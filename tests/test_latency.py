import json

import pytest
import torch

from tinyrouter import latency
from tinyrouter.latency import (
    ArchitectureMismatchError,
    haiku_latency,
    latency_summary,
    time_queries,
    trained_parameter_count,
)


def test_latency_summary_uses_linear_percentiles():
    stats = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean_ms"] == 2.5
    assert stats["p50_ms"] == 2.5
    # linear: rank 0.95 * 3 = 2.85 between 3 and 4
    assert stats["p95_ms"] == pytest.approx(3.85)
    assert (stats["min_ms"], stats["max_ms"]) == (1.0, 4.0)


def test_latency_summary_p95_is_not_the_max_on_a_long_list():
    stats = latency_summary([float(v) for v in range(1, 101)])
    assert stats["p50_ms"] == pytest.approx(50.5)
    assert stats["p95_ms"] == pytest.approx(95.05)


def test_latency_summary_refuses_an_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        latency_summary([])


def test_haiku_latency_splits_rows_and_counts_retries(tmp_path):
    rows = [
        {"split": "validation", "latency_ms": 100, "attempts": 1},
        {"split": "validation", "latency_ms": 300, "attempts": 2},
        {"split": "test", "latency_ms": 200, "attempts": 1},
    ]
    journal = tmp_path / "haiku.jsonl"
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = haiku_latency(journal)
    assert out["calls_retried"] == 1
    assert out["all"]["n"] == 3
    assert out["all"]["mean_ms"] == 200.0
    assert out["splits"]["validation"]["p50_ms"] == 200.0
    assert out["splits"]["test"]["n"] == 1
    assert "network" in out["measures"]


def test_trained_parameter_count_reads_the_k100_seed42_run():
    assert trained_parameter_count(latency.Path("results"), "modernbert") == 149_720_983
    assert trained_parameter_count(latency.Path("results"), "bert") == 109_598_359


class FakeTokenizer:
    def __call__(self, texts, truncation, max_length, return_tensors):
        assert truncation and return_tensors == "pt"
        ids = torch.tensor([[len(t) % 7 for t in texts[0][:max_length]]])
        return {"input_ids": ids}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 151)

    def forward(self, input_ids):
        return type("Out", (), {"logits": self.linear(input_ids.float().mean(1, keepdim=True))})


def test_time_queries_times_every_query_in_order_and_forward_is_not_longer():
    end_to_end, forward = time_queries(FakeModel(), FakeTokenizer(), ["ab", "cde", "f"], 8)
    assert len(end_to_end) == len(forward) == 3
    assert all(e >= f > 0 for e, f in zip(end_to_end, forward, strict=True))


def test_benchmark_stops_when_the_timed_model_is_not_the_trained_architecture(monkeypatch):
    monkeypatch.setattr(latency, "load_timed_model", lambda config: (FakeModel(), FakeTokenizer()))
    with pytest.raises(ArchitectureMismatchError, match="parameters"):
        latency.benchmark_encoder("bert", latency.Path("results"), ["a"], threads=1)


def test_committed_cpu_latency_records_method_hardware_and_matching_parameters():
    body = json.loads(latency.Path("results/efficiency/cpu_latency.json").read_text())
    assert body["method"]["batch_size"] == 1
    assert body["method"]["device"] == "cpu"
    assert body["hardware"]["cpu"]
    for name in ("bert", "modernbert"):
        model = body["models"][name]
        assert model["parameters_match_trained_run"] is True
        assert model["parameters_total"] == trained_parameter_count(latency.Path("results"), name)
        assert model["end_to_end"]["n"] == 500
