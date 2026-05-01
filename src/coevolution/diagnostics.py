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
punchline of the project: the latter climbs while the former plateaus or falls.

Each metric is exposed as a standalone pure function so it can be tested in
isolation with controlled inputs, then composed into compute_diagnostics for
use by the training loop.
"""

import logging

import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from coevolution.judge import Judge
from coevolution.policy import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _policy_probs(
    policy_params: dict,
    policy: Policy,
    outputs_tbl: jnp.ndarray,
) -> jnp.ndarray:
    """Return π(x) for every sequence in the enumerated output table.

    The autoregressive policy is a proper probability distribution, so the
    returned values sum to 1 (up to float32 rounding across 4096 entries).

    Args:
        policy_params: Flax parameter dict for the policy.
        policy: Policy module instance.
        outputs_tbl: All output sequences, shape (output_space, seq_len).

    Returns:
        Float array of shape (output_space,) containing π(x) for each x.
    """
    log_probs = policy.log_prob(policy_params, outputs_tbl)
    return jnp.exp(log_probs)


def _ensemble_judge_scores(
    judges_params: list[dict],
    judge: Judge,
    outputs_tbl: jnp.ndarray,
) -> jnp.ndarray:
    """Return mean judge score J(x) for every sequence, averaged across ensemble.

    For n_judges=1 (baseline) this reduces to a single judge.apply call.
    Averaging here mirrors what _compute_rewards does inside the training loop,
    so diagnostics reflect the same signal the policy was trained on.

    Args:
        judges_params: List of judge parameter dicts, one per ensemble member.
        judge: Judge module instance (shared architecture across members).
        outputs_tbl: All output sequences, shape (output_space, seq_len).

    Returns:
        Float array of shape (output_space,) containing J(x) for each x.
    """
    return jnp.stack(
        [judge.apply(p, outputs_tbl) for p in judges_params]
    ).mean(axis=0)


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def policy_quality(
    probs: jnp.ndarray,
    q_star_tbl: jnp.ndarray,
) -> float:
    """Compute E_x~π[q*(x)]: expected ground-truth quality under the policy.

    This is the metric the experiment ultimately cares about. It is not
    visible to the training loop — only the judge score is — which is why
    it can diverge from judge_perceived_quality as co-evolution progresses.

    Args:
        probs: Policy probabilities π(x) for all outputs, shape (output_space,).
            Must be non-negative and sum to approximately 1.
        q_star_tbl: Ground-truth quality q*(x) for all outputs, shape (output_space,).

    Returns:
        Scalar expected quality in [min(q*), max(q*)].
    """
    return float(jnp.sum(probs * q_star_tbl))


def judge_alignment(
    judge_scores: jnp.ndarray,
    q_star_tbl: jnp.ndarray,
) -> float:
    """Compute Spearman(J(x), q*(x)): rank correlation between judge and ground truth.

    Rank correlation rather than Pearson is used because the judge's score
    scale is arbitrary — only the ordering matters for the preference learning
    objective. A score of 1.0 means the judge ranks sequences identically to
    q*; 0.0 means the judge's ranking is uninformative.

    Args:
        judge_scores: Judge scores J(x) for all outputs, shape (output_space,).
        q_star_tbl: Ground-truth quality q*(x) for all outputs, shape (output_space,).

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    return float(spearmanr(np.array(judge_scores), np.array(q_star_tbl))[0])


def policy_entropy(probs: jnp.ndarray) -> float:
    """Compute H(π): policy entropy in nats.

    Entropy is tracked because distributional collapse — entropy declining
    monotonically to near-zero — is one of the three documented failure modes
    of self-rewarding training. A high-entropy policy explores broadly; a
    collapsed policy has converged on a few sequences the judge prefers.

    Args:
        probs: Policy probabilities π(x) for all outputs, shape (output_space,).
            Must be non-negative. Need not sum exactly to 1.

    Returns:
        Entropy H(π) = -Σ π(x) log π(x) in nats. Zero-probability entries
        contribute zero (0 * log(0) = 0 by convention).
    """
    # Use where to avoid passing zero to log: log(0) = -inf, and 0 * -inf = NaN
    # in JAX, which would corrupt the result. The substituted value (1.0) is
    # only used when probs == 0, where the outer probs * ... factor zeroes it out.
    safe_log = jnp.where(probs > 0, jnp.log(jnp.where(probs > 0, probs, 1.0)), 0.0)
    return float(-jnp.sum(probs * safe_log))


def judge_perceived_quality(
    probs: jnp.ndarray,
    judge_scores: jnp.ndarray,
) -> float:
    """Compute E_x~π[J(x)]: expected judge score under the policy.

    This is the quantity the REINFORCE objective actually maximises. It is
    expected to rise monotonically even after policy_quality plateaus, because
    the judge drifts to assign higher scores to whatever the policy produces
    — the reward hacking signature.

    Args:
        probs: Policy probabilities π(x) for all outputs, shape (output_space,).
        judge_scores: Judge scores J(x) for all outputs, shape (output_space,).

    Returns:
        Scalar expected judge score.
    """
    return float(jnp.sum(probs * judge_scores))


# ---------------------------------------------------------------------------
# Ceiling detection
# ---------------------------------------------------------------------------


def detect_ceiling(history: list[dict]) -> dict:
    """Summarise the ceiling and degradation effect over a completed run.

    The ceiling effect is the observation that policy_quality stops improving
    — or reverses — after a small number of iterations even as
    judge_perceived_quality continues to climb. This helper locates the peak
    and quantifies the subsequent degradation so it can be reported in the
    writeup without manual inspection of the JSONL output.

    Args:
        history: List of per-iteration diagnostic dicts produced by
            run_coevolution. Each dict must contain a "policy_quality" key.
            Must be non-empty.

    Returns:
        Dict with:
          - iteration_at_peak (int): 0-based index of the iteration with the
            highest policy_quality.
          - peak_quality (float): Highest policy_quality observed.
          - final_quality (float): policy_quality at the last iteration.
          - degradation_pct (float): 100 * (peak - final) / |peak|.
            Positive means quality fell after peaking; zero means quality
            never declined (no ceiling observed within these iterations).

    Raises:
        ValueError: If history is empty or any record lacks "policy_quality".
    """
    if not history:
        raise ValueError("history must be non-empty.")
    if "policy_quality" not in history[0]:
        raise ValueError(
            "Each record in history must contain a 'policy_quality' key. "
            "Ensure compute_diagnostics has been run (M4)."
        )

    qualities = [r["policy_quality"] for r in history]
    peak_idx = int(np.argmax(qualities))
    peak_q = qualities[peak_idx]
    final_q = qualities[-1]
    degradation = 100.0 * (peak_q - final_q) / (abs(peak_q) + 1e-8)

    return {
        "iteration_at_peak": peak_idx,
        "peak_quality": peak_q,
        "final_quality": final_q,
        "degradation_pct": degradation,
    }


# ---------------------------------------------------------------------------
# Orchestrating function called by the training loop
# ---------------------------------------------------------------------------


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
    """Compute all four headline metrics for one outer iteration.

    Called once per outer iteration by run_coevolution. All metrics are exact
    (enumerated over the full output space) so the results carry no sampling
    variance — every run with the same parameters produces identical numbers.

    Args:
        policy_params: Current policy Flax parameter dict.
        judges_params: List of current judge Flax parameter dicts, one per
            ensemble member (length 1 for the baseline).
        q_star_tbl: Precomputed q*(x) for all outputs, shape (output_space,).
        outputs_tbl: All output sequences, shape (output_space, seq_len).
        policy: Policy module instance.
        judge: Judge module instance.
        iteration: Current outer iteration index (0-based).
        seed: Run seed, included in every record for downstream joining.

    Returns:
        Dict with keys: iteration, seed, policy_quality, judge_alignment,
        policy_entropy, judge_perceived_quality.
    """
    probs = _policy_probs(policy_params, policy, outputs_tbl)
    j_scores = _ensemble_judge_scores(judges_params, judge, outputs_tbl)

    diag = {
        "iteration": iteration,
        "seed": seed,
        "policy_quality": policy_quality(probs, q_star_tbl),
        "judge_alignment": judge_alignment(j_scores, q_star_tbl),
        "policy_entropy": policy_entropy(probs),
        "judge_perceived_quality": judge_perceived_quality(probs, j_scores),
    }
    logger.debug("Iteration %d diagnostics: %s", iteration, diag)
    return diag
