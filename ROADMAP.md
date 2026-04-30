# ROADMAP

This repository is the empirical companion to my four-part research series, ["From Human Feedback to Synthetic Alignment"](https://jmamath.github.io/blog/synthetic-alignment-overview/) (Nov 2025). The series surveyed where the field stands and identified five open questions; this repository is an attempt to bring one of them — judge-policy co-evolution dynamics, [Frontier 5](https://jmamath.github.io/blog/synthetic-alignment-future/) — into a regime small enough to study cleanly.

## Research question

When a judge model and a policy co-evolve through self-rewarding training, three phenomena have been documented across recent work:

1. **The 3–4 iteration ceiling on self-improvement** (Yuan et al. 2025; Wu et al. 2024; Dong et al. 2024).
2. **Judge–ground-truth alignment drift**: as the policy distribution shifts, the judge's correlation with ground-truth quality decays even as judge-perceived quality keeps rising (Gao et al. 2022; Casper et al. 2023).
3. **Distributional collapse**: policy entropy declines monotonically across iterations.

Two questions:

- Do these phenomena reproduce at miniature scale — networks on the order of 5K parameters, on a synthetic task where ground-truth quality is computable by enumeration?
- If so, do the architectural mitigations discussed in Part 4 of the series (asymmetric update rates, judge ensembles, periodic reinitialization, external validation anchors) actually preserve quality, or do they shift the failure mode?

## Why toy scale

Toy scale lets us compute every diagnostic exactly. The policy is a categorical distribution over 4096 sequences, so policy quality `E_x~π[q*(x)]`, judge alignment `Spearman(J, q*)`, and policy entropy `H(π)` can all be computed by enumeration over the output space. The dynamics are the dynamics; they are not obscured by sampling noise.

The cost is that the experiment does not speak directly to LLM-scale phenomena. That is a real limitation, listed in the writeup. The point of this work is not to prove anything about LLMs — it is to make the dynamics visible and the architectural variants testable in a regime where what is happening can be measured precisely.

## Stack

- Python 3.11
- JAX (CPU) and Flax for models
- Optax for optimizers
- NumPy, SciPy (`spearmanr`), matplotlib
- pytest, ruff

JAX CPU on Apple Silicon is well-optimized via the Accelerate framework; no GPU required at this scale. A full baseline run completes in roughly ten minutes on an M1 laptop.

## Repository layout

```
.
├── README.md
├── ROADMAP.md                   # this file
├── pyproject.toml
├── src/
│   └── coevolution/
│       ├── __init__.py
│       ├── world.py             # output space + ground-truth quality function
│       ├── policy.py            # categorical autoregressive policy
│       ├── judge.py             # sequence scorer
│       ├── train.py             # co-evolution loop + variants
│       ├── diagnostics.py       # exact metrics over the enumerated output space
│       ├── config.py            # dataclass configs
│       └── utils.py             # PRNG, logging, checkpointing
├── tests/
├── scripts/
│   ├── run_baseline.py          # baseline 10-iteration run across 5 seeds
│   ├── run_variants.py          # asymmetric updates, ensembles, reinit, anchor
│   └── make_figures.py          # produces the headline 2×2 plot
└── results/                     # gitignored
```

## Milestones

The work is structured as six milestones, each with explicit acceptance criteria.

### M0 — Repository bootstrap

Pinned dependencies, ruff and pytest configuration, virtualenv ignored, README skeleton stating the research question.

*Acceptance:* `pip install -e ".[dev]"` succeeds; `pytest` runs; `ruff check .` is clean; JAX reports a CPU device.

### M1 — Toy world

Output space: sequences of length `L = 4` over a vocabulary of `K = 8` compass direction symbols — ↑ ↗ → ↘ ↓ ↙ ← ↖ — giving 8⁴ = 4096 possible sequences. A sequence looks like `↑ ↗ ↓ ←`.

Ground-truth quality `q*(x)` is a fixed compositional function combining per-position token weights with a small pair-interaction term. The full table over all 4096 outputs is precomputed and used as an oracle in diagnostics — it is *not* visible to the judge.

*Acceptance:* tests confirm `q*` is deterministic given a seed, covers all 4096 outputs, and has non-degenerate variance.

### M2 — Models

Policy: token-embedding → 1-layer GRU → linear projection to vocab logits, around 5K parameters. Generates by sampling at training time, by greedy decoding at analysis time.

Judge: token-embedding → 1-layer GRU → final hidden state → scalar score, around 5K parameters.

*Acceptance:* forward-pass shape tests pass; combined parameter count under 100K; both networks support sample and score paths.

### M3 — Co-evolution loop

The judge is initialized to *partial* alignment with `q*` — pretrained until Spearman correlation with `q*` reaches `judge_init_alignment` (default 0.7). This is the regime where co-evolution is interesting; random init or perfect init are degenerate cases.

Each outer iteration runs:

1. Sample paired outputs from the current policy.
2. Compute judge preferences within pairs.
3. Update the policy via REINFORCE with judge score as reward (with an entropy bonus).
4. Update the judge on the policy's own (judge-labeled) preference data.

Defaults: 10 outer iterations, 200 inner steps per iteration, batch 64, learning rates 3e-4 for both networks, entropy bonus 0.01, seeds 0–4.

*Acceptance:* a single iteration runs without NaN gradients; a full 10-iteration run completes in under ten minutes on M1 CPU; all four diagnostics are saved per iteration.

### M4 — Diagnostics

Four metrics, each computed exactly by enumeration over the output space:

- `policy_quality(π) = E_x~π[q*(x)]`
- `judge_alignment(J) = Spearman(J(x), q*(x))` over uniformly sampled `x`
- `policy_entropy(π) = H(π)`
- `judge_perceived_quality(π, J) = E_x~π[J(x)]`

A small ceiling-detection helper records `iteration_at_peak`, `peak_quality`, `final_quality`, and `degradation_pct`.

*Acceptance:* identity, random, and aligned test cases produce the expected metric values; metrics are saved per outer iteration in JSONL.

### M5 — Architectural variants

Four mitigations, each a single config flag changing the judge-update step:

- **Asymmetric update rates**: `judge_update_every ∈ {1, 2, 5, 10}`.
- **Judge ensembles**: `n_judges ∈ {1, 3, 5}`, with policy reward = mean ensemble score.
- **Periodic reinitialization**: `reset_every ∈ {∞, 3, 5}` outer iterations.
- **External validation anchor**: `anchor_fraction ∈ {0.0, 0.05, 0.1}` of judge-update batches drawn uniformly from the output space and labeled with `q*`.

Each variant is run across 5 seeds. Total: roughly 65 runs at ~10 minutes each, parallelizable across the M1's cores.

*Acceptance:* per-variant CSVs and plots saved; the writeup section on which variants helped is grounded in the data and honest about negative results.

### M6 — Writeup

Blog post and README. The post follows: setup → headline finding (2×2 figure) → variants finding → connection back to the prior series → limitations → code link.

The headline figure is a 2×2 panel sharing the iteration axis: top-left `policy_quality`, top-right `judge_alignment`, bottom-left `policy_entropy`, bottom-right `judge_perceived_quality`. Mean line plus standard-deviation band over five seeds. The visual punchline is the contrast between bottom-right (looks like progress) and top-left (the truth).

## Reproducing the headline result

```bash
git clone <repo-url>
cd <repo-name>
pip install -e ".[dev]"
python scripts/run_baseline.py --seeds 0 1 2 3 4
python scripts/make_figures.py --runs results/baseline_seeds_0-4
```

Expected wall time on an M1 laptop: ten to fifteen minutes for all five seeds; under a minute for the figures. Output: `results/baseline_seeds_0-4/headline_2x2.png`.

## Scope and non-goals

Out of scope for this version of the project:

- Real language models. The phenomena under study are well documented at LLM scale; this work is about making the dynamics visible at a scale where they can be measured exactly, not about pushing scale.
- Conditional generation. The toy is unconditional; adding prompts does not change the dynamics being studied and does add analysis complexity.
- Hyperparameter optimization beyond the ablation grid. The point is reproducing a qualitative phenomenon, not topping any benchmark.
- DPO from scratch. REINFORCE with judge-as-reward exposes the dynamics most directly; DPO comparisons are reserved as a stretch.

## Stretch directions

For follow-up work after the headline result is reproduced:

- A small real-LM variant using Pythia-160M or Qwen2.5-0.5B with LoRA. Same loop, same diagnostics, retaining the toy `q*` so ground truth remains tractable.
- A meta-judge module updating the judge based on its accuracy against a held-out anchor set — the simplest empirical version of Wu et al. (2024) at toy scale.
- DPO vs. REINFORCE comparison on the policy update.

## References

The series this work continues:

- *From Human Feedback to Synthetic Alignment* — [Part 1](https://jmamath.github.io/blog/rlhf-limitations/), [Part 2](https://jmamath.github.io/blog/synthetic-alignment-architecture/), [Part 3](https://jmamath.github.io/blog/what-works-synthetic-alignment/), [Part 4](https://jmamath.github.io/blog/synthetic-alignment-future/), [overview](https://jmamath.github.io/blog/synthetic-alignment-overview/).

The papers this work directly draws on:

- Yuan, W., Pang, R. Y., Cho, K., et al. (2025). *Self-Rewarding Language Models.* arXiv:2401.10020.
- Wu, T., Yuan, W., Golovneva, O., et al. (2024). *Meta-Rewarding Language Models.* arXiv:2407.19594.
- Dong, Q., Dong, L., Zhang, X., Sui, Z., Wei, F. (2024). *Self-Boosting Large Language Models with Synthetic Preference Data.* arXiv:2410.06961.
- Guo, S., Zhang, B., Liu, T., et al. (2024). *Direct Language Model Alignment from Online AI Feedback.* arXiv:2402.04792.
- Gao, L., Schulman, J., Hilton, J. (2022). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760.
- Casper, S., Davies, X., Shi, C., et al. (2023). *Open Problems and Fundamental Limitations of RLHF.* arXiv:2307.15217.
