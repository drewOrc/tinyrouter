"""``make figures``: the four README figures, drawn from committed JSON only.

Reads ``results/analysis/summary.json`` and ``curves.json`` and writes PNGs
(300 dpi) to ``results/figures/``. Nothing is random, and the PNG
``Software`` tag (the matplotlib version) is dropped, so two runs with the
same matplotlib give byte-identical files. Colours are the Okabe-Ito
palette, which stays distinguishable under the common colour-vision
deficiencies; line styles and markers differ too, so the figures also work
in greyscale. Bands and error bars are mean ± sample std over seeds 42, 43, 44.

Each seed's final router uses the 8-way aggregation it chose on validation.
In risk_coverage a k whose seeds chose differently gets one panel per
aggregation, titled with the seeds that use it; nothing is chosen on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KS = (1, 5, 10, 25, 50, 100)
TARGET = "0.02"
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
STYLE = {
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
MODELS = {
    "modernbert": ("ModernBERT-base", OKABE_ITO["blue"], "-", "o"),
    "bert": ("BERT-base", OKABE_ITO["vermillion"], "--", "s"),
    "tfidf-centroid": ("TF-IDF centroid", OKABE_ITO["green"], ":", "^"),
}
SIGNALS = {
    "msp_t": ("MSP, temperature-scaled", OKABE_ITO["blue"], "-", 1.8),
    "msp": ("MSP", OKABE_ITO["sky"], "--", 1.0),
    "entropy": ("negative entropy", OKABE_ITO["orange"], "-.", 1.0),
    "margin": ("top-2 margin", OKABE_ITO["purple"], ":", 1.2),
}


def load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    return plt


def final(summary: dict, group: str) -> dict:
    return summary["groups"][group]["result"]["final"]


def series(summary: dict, model: str, metric: str) -> tuple[list[float], list[float]]:
    """Per-k (mean, std) in percent of the small-only ``metric``."""
    stats = [final(summary, f"{model}/k{k}")["small_only"][metric] for k in KS]
    return [100 * s["mean"] for s in stats], [100 * (s["std"] or 0.0) for s in stats]


def log_k_axis(ax) -> None:  # noqa: ANN001
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlabel("labelled examples per intent, k (log scale)")


def learning_curves(summary: dict, plt) -> object:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), layout="constrained")
    haiku = summary["llm_only_test"]
    panels = (
        ("accuracy_8", "8-way routing accuracy (%)", haiku["accuracy_8"]),
        ("oos_recall", "OOS recall (%)", haiku["oos"]["recall"]),
    )
    for ax, (metric, ylabel, haiku_value) in zip(axes, panels, strict=True):
        ax.axvspan(0.85, 10 * 1.2, color=OKABE_ITO["grey"], alpha=0.15, lw=0)
        for model, (label, colour, style, marker) in MODELS.items():
            mean, std = series(summary, model, metric)
            ax.plot(KS, mean, style, color=colour, marker=marker, ms=3.5, lw=1.4, label=label)
            ax.fill_between(
                KS,
                [m - s for m, s in zip(mean, std, strict=True)],
                [m + s for m, s in zip(mean, std, strict=True)],
                color=colour,
                alpha=0.2,
                lw=0,
            )
        ax.axhline(
            100 * haiku_value,
            color=OKABE_ITO["black"],
            ls="-.",
            lw=1.0,
            label="Claude Haiku 4.5, zero-shot",
        )
        ax.text(
            1.0,
            0.98 if metric == "accuracy_8" else 0.9,
            "k ≤ 10: fixed\n400-step budget",
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            va="top",
        )
        log_k_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
    axes[0].set_title("(a) 8-way accuracy, test")
    axes[1].set_title("(b) OOS recall, test")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False
    )
    return fig


AGGREGATIONS = ("argmax", "summed")
SEEDS = (42, 43, 44)


def seed_aggregations(summary: dict, group: str) -> tuple[str, ...]:
    """The 8-way aggregation each seed's final router chose on validation, in seed order.

    ``summary.json`` stores one string when the three seeds agree and the
    per-seed list when they do not (``analysis_run.combine``).
    """
    chosen = summary["groups"][group]["result"]["selected_aggregation"]
    if isinstance(chosen, str):
        return (chosen,) * len(SEEDS)
    if len(chosen) != len(SEEDS) or not set(chosen) <= set(AGGREGATIONS):
        raise ValueError(f"{group}: unexpected selected_aggregation {chosen!r}")
    return tuple(chosen)


def risk_coverage_panels(summary: dict) -> list[tuple[int, str, tuple[int, ...] | None]]:
    """(k, aggregation, seeds whose final router uses it) per panel, None when all three do.

    A k whose seeds agree gets one panel. A k whose seeds chose different
    aggregations on validation gets one panel per aggregation that a seed
    chose, in the fixed order of ``AGGREGATIONS``; none is picked on test
    and none is dropped.
    """
    panels: list[tuple[int, str, tuple[int, ...] | None]] = []
    for k in (10, 100):
        per_seed = seed_aggregations(summary, f"modernbert/k{k}")
        if len(set(per_seed)) == 1:
            panels.append((k, per_seed[0], None))
            continue
        for aggregation in AGGREGATIONS:
            seeds = tuple(s for s, a in zip(SEEDS, per_seed, strict=True) if a == aggregation)
            if seeds:
                panels.append((k, aggregation, seeds))
    return panels


def panel_title(index: int, k: int, aggregation: str, seeds: tuple[int, ...] | None) -> str:
    title = f"({'abcd'[index]}) ModernBERT k={k} ({aggregation})"
    if seeds is None:
        return title
    plural = "s" * (len(seeds) > 1)
    return f"{title}\nmean of 3 seeds; final router of seed{plural} {', '.join(map(str, seeds))}"


def risk_coverage(summary: dict, curves: dict, plt) -> object:  # noqa: ANN001
    """Test risk-coverage of each signal; a k with mixed per-seed aggregations gets two panels.

    ``curves.json`` keeps each aggregation's curve as mean and std over all
    three seeds, not per seed, so a mixed k shows both aggregations side
    by side, each titled with the seeds whose final router uses it.
    """
    panels = risk_coverage_panels(summary)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.4 * len(panels), 3.0), sharey=True, layout="constrained"
    )
    grid = [100 * c for c in curves["coverage_grid"]]
    for index, (ax, (k, aggregation, seeds)) in enumerate(zip(axes, panels, strict=True)):
        block = curves["groups"][f"modernbert/k{k}"]["result"][aggregation]
        for signal, (label, colour, style, width) in SIGNALS.items():
            points = block[signal]["risk_coverage"]
            mean = [100 * p["mean"] for p in points]
            ax.plot(grid, mean, style, color=colour, lw=width, label=label)
            if signal == "msp_t":
                std = [100 * (p["std"] or 0.0) for p in points]
                ax.fill_between(
                    grid,
                    [m - s for m, s in zip(mean, std, strict=True)],
                    [m + s for m, s in zip(mean, std, strict=True)],
                    color=colour,
                    alpha=0.2,
                    lw=0,
                )
        ax.axhline(
            100 * float(TARGET),
            color=OKABE_ITO["grey"],
            ls="--",
            lw=0.9,
            label=f"{100 * float(TARGET):.0f}% target risk",
        )
        ax.set_title(panel_title(index, k, aggregation, seeds))
        ax.set_xlabel("coverage: queries kept by the small model (%)")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("selective risk on test (%)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False
    )
    return fig


def router_stats(summary: dict, k: int) -> dict[str, tuple[float, float, float, float]]:
    """Router -> (accuracy mean, std, call-rate mean, std) in percent."""
    haiku = summary["llm_only_test"]
    block = final(summary, f"modernbert/k{k}")
    metrics = {
        "small-only": block["small_only"],
        f"hybrid ({100 * float(TARGET):.0f}% target)": block["fallback"][TARGET]["hybrid"]["test"],
        "oracle": block["oracle"],
    }
    out = {"LLM-only (Haiku)": (100 * haiku["accuracy_8"], 0.0, 100.0, 0.0)}
    for name, m in metrics.items():
        acc, calls = m["accuracy_8"], m["llm_call_rate"]
        out[name] = (
            100 * acc["mean"],
            100 * (acc["std"] or 0.0),
            100 * calls["mean"],
            100 * (calls["std"] or 0.0),
        )
    return out


ROUTER_STYLE = (
    (OKABE_ITO["grey"], "D", ""),
    (OKABE_ITO["sky"], "s", "//"),
    (OKABE_ITO["blue"], "o", ""),
    (OKABE_ITO["orange"], "^", ".."),
)
ROUTER_OFFSET = 0.2


def accuracy_dots(ax, stats: dict, names: list[str]) -> None:  # noqa: ANN001
    """Panel (a): points with error bars, so the y range does not turn into bar lengths."""
    for i, name in enumerate(names):
        colour, marker, _ = ROUTER_STYLE[i]
        xs = [j + (i - 1.5) * ROUTER_OFFSET for j in range(2)]
        means = [stats[k][name][0] for k in (10, 100)]
        errors = [stats[k][name][1] for k in (10, 100)]
        ax.errorbar(
            xs,
            means,
            yerr=errors,
            fmt=marker,
            color=colour,
            mec="black",
            mew=0.5,
            ms=6,
            capsize=2,
            lw=1.0,
            label=name,
        )
        for x, m in zip(xs, means, strict=True):
            ax.annotate(
                f"{m:.1f}",
                (x, m),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )
    ax.set_ylabel("8-way accuracy on test (%)")
    ax.set_ylim(75, 100)
    ax.grid(axis="y", color=OKABE_ITO["grey"], alpha=0.3, lw=0.5)


def call_bars(ax, stats: dict, names: list[str]) -> None:  # noqa: ANN001
    """Panel (b): bars from 0, since the call rate is a share of all queries."""
    for i, name in enumerate(names):
        colour, _, hatch = ROUTER_STYLE[i]
        xs = [j + (i - 1.5) * ROUTER_OFFSET for j in range(2)]
        bars = ax.bar(
            xs,
            [stats[k][name][2] for k in (10, 100)],
            ROUTER_OFFSET,
            yerr=[stats[k][name][3] for k in (10, 100)],
            capsize=2,
            color=colour,
            hatch=hatch,
            edgecolor="black",
            lw=0.5,
            label=name,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=1)
    ax.set_ylabel("queries sent to Haiku (%)")
    ax.set_ylim(0, 110)


def routers(summary: dict, plt) -> object:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), layout="constrained")
    stats = {k: router_stats(summary, k) for k in (10, 100)}
    names = list(stats[10])
    accuracy_dots(axes[0], stats, names)
    call_bars(axes[1], stats, names)
    for ax in axes:
        ax.set_xticks([0, 1], ["ModernBERT k=10", "ModernBERT k=100"])
        ax.set_xlim(-0.5, 1.5)
    axes[0].set_title("(a) accuracy (points; axis from 75%)")
    axes[1].set_title("(b) LLM call rate")
    handles = axes[0].get_legend_handles_labels()
    fig.legend(*handles, loc="outside lower center", ncol=4, frameon=False)
    return fig


def transfer_series(summary: dict, model: str) -> dict[str, list[tuple[int, float, float]]]:
    """Per feasible k: (k, mean, std) in percent of val, test and reweighted-test risk."""
    keys = {
        "validation": ("validation", "selective_risk"),
        "test": ("test", "selective_risk"),
        "reweighted": ("test_reweighted_to_validation_oos_share", "selective_risk"),
    }
    out: dict[str, list[tuple[int, float, float]]] = {name: [] for name in keys}
    for k in KS:
        diag = summary["groups"][f"{model}/k{k}"]["result"]["diagnostics"][TARGET]
        if diag is None:
            continue
        for name, (block, metric) in keys.items():
            stat = diag[block][metric]
            if stat is None or stat.get("n") is not None:
                continue
            out[name].append((k, 100 * stat["mean"], 100 * (stat["std"] or 0.0)))
    return out


def threshold_transfer(summary: dict, plt) -> object:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True, layout="constrained")
    lines = {
        "validation": ("validation (where the threshold was chosen)", OKABE_ITO["sky"], "--", "o"),
        "test": ("test", OKABE_ITO["vermillion"], "-", "s"),
        "reweighted": ("test reweighted to the validation OOS share", OKABE_ITO["green"], ":", "^"),
    }
    for ax, model in zip(axes, ("modernbert", "bert"), strict=True):
        for name, points in transfer_series(summary, model).items():
            label, colour, style, marker = lines[name]
            ks, mean, std = zip(*points, strict=True)
            ax.errorbar(
                ks,
                mean,
                yerr=std,
                fmt=style,
                color=colour,
                marker=marker,
                ms=3.5,
                lw=1.3,
                capsize=2,
                label=label,
            )
        ax.axhline(
            100 * float(TARGET),
            color=OKABE_ITO["black"],
            ls="-.",
            lw=0.9,
            label=f"{100 * float(TARGET):.0f}% target",
        )
        log_k_axis(ax)
        ax.set_title(f"({'ab'[model == 'bert']}) {MODELS[model][0]}")
    axes[0].set_ylabel("selective risk of the hybrid (%)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False
    )
    return fig


def build(root: Path, out: Path) -> list[Path]:
    plt = pyplot()
    summary = load(root, "analysis/summary.json")
    curves = load(root, "analysis/curves.json")
    figures = {
        "learning_curves.png": learning_curves(summary, plt),
        "risk_coverage.png": risk_coverage(summary, curves, plt),
        "routers.png": routers(summary, plt),
        "threshold_transfer.png": threshold_transfer(summary, plt),
    }
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fig in figures.items():
        path = out / name
        fig.savefig(path, format="png", metadata={"Software": None})
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default=None, help="default: <results-root>/figures")
    args = parser.parse_args(argv)
    root = Path(args.results_root)
    for path in build(root, Path(args.out) if args.out else root / "figures"):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
