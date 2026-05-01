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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, required=True, help="Directory containing JSONL outputs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    runs_dir = pathlib.Path(args.runs)
    if not runs_dir.is_dir():
        logger.error("Directory %s does not exist.", runs_dir)
        return

    # Load all jsonl files
    all_data = [] # list of lists of dicts
    for file_path in runs_dir.glob("*.jsonl"):
        with open(file_path, "r") as f:
            run_data = [json.loads(line) for line in f]
            all_data.append(run_data)

    if not all_data:
        logger.error("No JSONL files found in %s", runs_dir)
        return

    n_iterations = len(all_data[0])
    iterations = np.arange(n_iterations)

    metrics = [
        "policy_quality",
        "judge_alignment",
        "policy_entropy",
        "judge_perceived_quality",
    ]

    metric_stats = {}
    for metric in metrics:
        # shape: (n_seeds, n_iterations)
        data_matrix = np.array([[row[metric] for row in run] for run in all_data])
        metric_stats[metric] = {
            "mean": data_matrix.mean(axis=0),
            "std": data_matrix.std(axis=0),
        }

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
        mean = metric_stats[metric]["mean"]
        std = metric_stats[metric]["std"]
        ax.plot(iterations, mean, color="blue", linewidth=2)
        ax.fill_between(iterations, mean - std, mean + std, color="blue", alpha=0.2)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        if row == 1:
            ax.set_xlabel("Iteration")

    # Annotate peak ground-truth quality
    peak_idx = np.argmax(metric_stats["policy_quality"]["mean"])
    axs[0, 0].axvline(x=peak_idx, color="red", linestyle="--", alpha=0.5)
    
    # Place text safely
    y_min, y_max = axs[0, 0].get_ylim()
    axs[0, 0].text(peak_idx + 0.2, y_min + 0.05 * (y_max - y_min), f"Peak: iter {peak_idx}", color="red")

    plt.tight_layout()
    out_path = runs_dir / "headline_2x2.png"
    plt.savefig(out_path, dpi=300)
    logger.info("Saved figure to %s", out_path)

if __name__ == "__main__":
    main()
