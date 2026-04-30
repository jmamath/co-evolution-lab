"""
Baseline co-evolution experiment across multiple seeds.

This script exists to produce the headline result: ten iterations of vanilla
self-rewarding training run across five independent seeds, with diagnostics
saved at each iteration. The resulting data feeds directly into make_figures.py
to produce the 2x2 headline plot.

Run:
    python scripts/run_baseline.py --seeds 0 1 2 3 4

Output:
    results/baseline_seeds_<range>/  — one JSONL file per seed, plus a
    combined CSV used by make_figures.py.

Expected wall time on M1: 10-15 minutes for all five seeds.
"""

import logging

# TODO: Milestone 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
