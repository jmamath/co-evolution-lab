"""
Exact diagnostic metrics computed by enumeration over the full output space.

This file exists to measure what is actually happening during co-evolution, as
opposed to what the training loop thinks is happening. All four metrics are
computed exactly — no sampling noise — because the output space (4096 sequences)
is fully enumerable. This is the key advantage of toy scale: the gap between
judge-perceived progress and ground-truth progress can be measured precisely
rather than estimated.

The four headline metrics are:
  - policy_quality:          E_x~π[q*(x)]          — true quality under policy
  - judge_alignment:         Spearman(J(x), q*(x))  — judge's fidelity to ground truth
  - policy_entropy:          H(π)                   — distributional spread
  - judge_perceived_quality: E_x~π[J(x)]            — what the loop optimises

The contrast between policy_quality and judge_perceived_quality is the visual
punchline of the project: the latter climbs while the former plateaus.
"""

import logging

import jax.numpy as jnp

from coevolution.judge import Judge
from coevolution.policy import Policy

logger = logging.getLogger(__name__)


def compute_diagnostics(
    policy_params: dict,
    judges_params: list[dict],
    q_star_tbl: jnp.ndarray,
    outputs_tbl: jnp.ndarray,
    policy: Policy,
    judge: Judge,
    iteration: int,
    seed: int,
) -> dict:
    """Compute per-iteration diagnostics over the full output space.

    This stub returns only iteration metadata. The four headline metrics
    (policy_quality, judge_alignment, policy_entropy, judge_perceived_quality)
    are implemented in Milestone 4.

    Args:
        policy_params: Current policy Flax parameter dict.
        judges_params: List of current judge Flax parameter dicts (one per
            ensemble member; length 1 for the baseline).
        q_star_tbl: Precomputed q*(x) for all outputs, shape (output_space,).
        outputs_tbl: All output sequences, shape (output_space, seq_len).
        policy: Policy module instance.
        judge: Judge module instance.
        iteration: Current outer iteration index (0-based).
        seed: Run seed, included in every record for traceability.

    Returns:
        Dict with at minimum {"iteration": int, "seed": int}. M4 extends
        this with the four headline metrics.
    """
    # TODO: Milestone 4 — replace this stub with exact metric computations
    return {"iteration": iteration, "seed": seed}
