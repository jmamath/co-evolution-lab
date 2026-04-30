"""
Architectural variant experiments.

This script exists to test the four stabilisation mechanisms proposed in
Frontier 5 of the blog series: asymmetric update rates, judge ensembles,
periodic reinitialization, and an external validation anchor. Each variant
changes a single config flag against the baseline; all other settings are
held constant so the effect is attributable to that mechanism alone.

Run:
    python scripts/run_variants.py

Output:
    results/variants/  — one subdirectory per variant, each containing
    per-seed JSONL files and a summary CSV.

Expected wall time on M1: 2-3 hours total; runs are parallelisable across cores.
"""

import logging

# TODO: Milestone 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
