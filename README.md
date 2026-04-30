# frontier-5-miniature

> *In November 2025 I argued judge-policy co-evolution dynamics were an open question in synthetic alignment. This is the smallest reproduction I could build of the phenomenon, and what I observed.*

This repository is the empirical companion to my four-part research series [*From Human Feedback to Synthetic Alignment*](https://jmamath.github.io/blog/synthetic-alignment-overview/). The series identified five open frontiers; this project brings [Frontier 5](https://jmamath.github.io/blog/synthetic-alignment-future/) — judge-policy co-evolution dynamics — into a regime small enough to measure precisely.

## Research question

When a judge model and a policy co-evolve through self-rewarding training, three phenomena have been documented:

1. **The 3–4 iteration ceiling on self-improvement** (Yuan et al. 2025; Wu et al. 2024)
2. **Judge–ground-truth alignment drift**: judge-perceived quality climbs while real quality plateaus, and judge correlation with ground truth decays
3. **Distributional collapse**: policy entropy declines monotonically

Do these reproduce at miniature scale (≈5K-parameter networks, synthetic task, ground truth computable by enumeration)? And do the proposed mitigations — asymmetric update rates, judge ensembles, periodic reinitialization, external validation anchors — actually help?

## Reproduce the headline result

```bash
git clone https://github.com/jmamath/frontier-5-miniature
cd frontier-5-miniature
pip install -e ".[dev]"
python scripts/run_baseline.py --seeds 0 1 2 3 4
python scripts/make_figures.py --runs results/baseline_seeds_0-4
```

Expected wall time on M1: ~10–15 minutes for all five seeds. Output: `results/baseline_seeds_0-4/headline_2x2.png`.

## Repository layout

```
.
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── src/
│   └── coevolution/
│       ├── world.py          # output space + ground-truth quality function
│       ├── policy.py         # categorical autoregressive policy
│       ├── judge.py          # sequence scorer
│       ├── train.py          # co-evolution loop + variants
│       ├── diagnostics.py    # exact metrics over enumerated output space
│       ├── config.py         # dataclass configs
│       └── utils.py          # PRNG, logging, checkpointing
├── tests/
├── scripts/
│   ├── run_baseline.py       # baseline 10-iteration run across 5 seeds
│   ├── run_variants.py       # asymmetric updates, ensembles, reinit, anchor
│   └── make_figures.py       # produces the headline 2×2 plot
└── results/                  # gitignored
```

## Stack

Python 3.11 · JAX (CPU) · Flax · Optax · NumPy · SciPy · Matplotlib · pytest · ruff

## References

- Yuan et al. (2025). *Self-Rewarding Language Models.* arXiv:2401.10020
- Wu et al. (2024). *Meta-Rewarding Language Models.* arXiv:2407.19594
- Dong et al. (2024). *Self-Boosting LLMs with Synthetic Preference Data.* arXiv:2410.06961
- Guo et al. (2024). *Direct LM Alignment from Online AI Feedback.* arXiv:2402.04792
- Gao, Schulman & Hilton (2022). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760
- Casper et al. (2023). *Open Problems and Fundamental Limitations of RLHF.* arXiv:2307.15217
