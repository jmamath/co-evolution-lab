"""
Figure generation for the headline result and variant comparisons.

This script exists to turn the raw JSONL/CSV output from run_baseline.py and
run_variants.py into publication-ready figures. It is intentionally separate
from the training scripts so figures can be regenerated (e.g. with different
styling) without re-running experiments.

Run:
    python scripts/make_figures.py --runs results/baseline_seeds_0-4

Output:
    results/baseline_seeds_0-4/headline_2x2.png  — the four-panel figure that
    is the centrepiece of the blog post and README.

The headline figure is a 2x2 grid sharing the iteration axis (0-10):
  top-left:     policy_quality          (E_x~π[q*(x)])
  top-right:    judge_alignment         (Spearman(J, q*))
  bottom-left:  policy_entropy          (H(π))
  bottom-right: judge_perceived_quality (E_x~π[J(x)])
Mean line + std band over seeds. The visual contrast between bottom-right
(looks like progress) and top-left (the ground truth) is the punchline.
"""

import argparse
import json
import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt

def load_run_data(runs_dir: pathlib.Path):
    """Load all JSONL files from a directory."""
    all_data = []
    for file_path in runs_dir.glob("*.jsonl"):
        with open(file_path, "r") as f:
            run_data = [json.loads(line) for line in f]
            all_data.append(run_data)
    return all_data


def get_stats(all_data, metric):
    """Compute mean and std for a metric across runs."""
    if not all_data:
        return None
    data_matrix = np.array([[row[metric] for row in run] for run in all_data])
    return {
        "mean": data_matrix.mean(axis=0),
        "std": data_matrix.std(axis=0),
    }


def plot_headline(runs_dir: pathlib.Path, out_path: pathlib.Path):
    """Produce the 2x2 headline figure."""
    all_data = load_run_data(runs_dir)
    if not all_data:
        return

    n_iterations = len(all_data[0])
    iterations = np.arange(n_iterations)

    metrics = [
        "policy_quality",
        "judge_alignment",
        "policy_entropy",
        "judge_perceived_quality",
    ]

    metric_stats = {m: get_stats(all_data, m) for m in metrics}

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig.suptitle("Judge-Policy Co-evolution Dynamics (Toy Scale)", fontsize=16)

    panels = [
        (0, 0, "policy_quality", "Policy Quality $\mathbb{E}[q^*(x)]$"),
        (0, 1, "judge_alignment", "Judge Alignment (Spearman)"),
        (1, 0, "policy_entropy", "Policy Entropy $\mathcal{H}(\pi)$"),
        (1, 1, "judge_perceived_quality", "Judge-Perceived Quality $\mathbb{E}[J(x)]$"),
    ]

    for row, col, metric, title in panels:
        ax = axs[row, col]
        stats = metric_stats[metric]
        ax.plot(iterations, stats["mean"], color="blue", linewidth=2)
        ax.fill_between(
            iterations,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="blue",
            alpha=0.2,
        )
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        if row == 1:
            ax.set_xlabel("Iteration")

    # Annotate peak ground-truth quality
    peak_idx = np.argmax(metric_stats["policy_quality"]["mean"])
    axs[0, 0].axvline(x=peak_idx, color="red", linestyle="--", alpha=0.5)
    y_min, y_max = axs[0, 0].get_ylim()
    axs[0, 0].text(
        peak_idx + 0.2,
        y_min + 0.05 * (y_max - y_min),
        f"Peak: iter {peak_idx}",
        color="red",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_comparison(
    variant_id: str,
    baseline_dir: pathlib.Path,
    variant_results: list[tuple[str, pathlib.Path]],
    out_path: pathlib.Path,
):
    """Plot a variant's settings against the baseline."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Stabilization Variant: {variant_id.replace('_', ' ').capitalize()}",
        fontsize=16,
    )

    metrics = [
        ("policy_quality", "Policy Quality"),
        ("judge_alignment", "Judge Alignment"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_results) + 1))

    baseline_data = load_run_data(baseline_dir)

    for i, (metric_key, title) in enumerate(metrics):
        ax = axs[i]
        # Plot baseline
        b_stats = get_stats(baseline_data, metric_key)
        iters = np.arange(len(b_stats["mean"]))
        ax.plot(
            iters,
            b_stats["mean"],
            label="Baseline",
            color=colors[0],
            linewidth=2,
            zorder=10,
        )
        ax.fill_between(
            iters,
            b_stats["mean"] - b_stats["std"],
            b_stats["mean"] + b_stats["std"],
            color=colors[0],
            alpha=0.1,
        )

        # Plot each setting of this variant
        for j, (label, runs_dir) in enumerate(variant_results):
            v_data = load_run_data(runs_dir)
            v_stats = get_stats(v_data, metric_key)
            if not v_stats:
                continue
            ax.plot(
                iters,
                v_stats["mean"],
                label=label,
                color=colors[j + 1],
                linewidth=1.5,
                alpha=0.8,
            )
            ax.fill_between(
                iters,
                v_stats["mean"] - v_stats["std"],
                v_stats["mean"] + v_stats["std"],
                color=colors[j + 1],
                alpha=0.05,
            )

        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, help="Baseline runs directory.")
    parser.add_argument("--variants", type=str, help="Variants root directory.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if args.runs:
        baseline_dir = pathlib.Path(args.runs)
        plot_headline(baseline_dir, baseline_dir / "headline_2x2.png")
        logger.info("Headline plot saved to %s", baseline_dir / "headline_2x2.png")

    if args.variants and args.runs:
        variants_root = pathlib.Path(args.variants)
        baseline_dir = pathlib.Path(args.runs)

        variant_groups = {}
        for d in variants_root.iterdir():
            if not d.is_dir():
                continue
            # name format: variantId_value
            parts = d.name.split("_")
            if len(parts) < 2:
                continue
            v_id = parts[0]
            v_val = parts[1]
            if v_id not in variant_groups:
                variant_groups[v_id] = []
            variant_groups[v_id].append((f"{v_id}={v_val}", d))

        for v_id, group in variant_groups.items():
            # Sort by value for legend order
            group.sort(key=lambda x: x[0])
            out_path = variants_root / f"comparison_{v_id}.png"
            plot_comparison(v_id, baseline_dir, group, out_path)
            logger.info("Comparison plot for %s saved to %s", v_id, out_path)


if __name__ == "__main__":
    main()
