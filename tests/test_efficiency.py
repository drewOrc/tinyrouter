import pytest

from tinyrouter.efficiency import (
    MEASURES,
    MemorySampler,
    count_parameters,
    memory_report,
    peak_rss_bytes,
)


def test_sampler_keeps_the_maximum_of_each_key_across_samples():
    readings = iter(
        [
            {"driver_allocated_bytes": 100, "tensor_allocated_bytes": 50},
            {"driver_allocated_bytes": 300, "tensor_allocated_bytes": 20},
            {"driver_allocated_bytes": 200, "tensor_allocated_bytes": 80},
        ]
    )
    sampler = MemorySampler(lambda: next(readings))
    for _ in range(3):
        sampler.sample()
    assert sampler.peaks == {"driver_allocated_bytes": 300, "tensor_allocated_bytes": 80}
    report = memory_report("mps", sampler)
    assert report["samples"] == 3
    assert report["measures"] == MEASURES["mps"]


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 2048), ("linux", 2048 * 1024)])
def test_ru_maxrss_unit_differs_between_macos_and_linux(platform, expected):
    assert peak_rss_bytes(2048, platform) == expected


def test_count_parameters_separates_frozen_ones():
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    assert count_parameters(model) == {"total": 12 + 4 + 8 + 2, "trainable": 4 + 8 + 2}
