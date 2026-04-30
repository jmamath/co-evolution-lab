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

import logging

# TODO: Milestone 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
