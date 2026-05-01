"""
Co-evolution training loop and architectural variant implementations.

This file exists to orchestrate the self-rewarding loop that is the subject of
the experiment. A single outer iteration runs: (1) sample from policy,
(2) judge scores the samples, (3) policy is updated via REINFORCE with judge
score as reward, (4) judge is updated on policy-generated pairs with its own
labels as supervision.

Each of the five architectural variants is isolated in its own private helper:

- _compute_rewards        — Variant B: ensemble mean vs. single-judge score
- _update_single_judge    — Variant D: anchor pairs mixed into the judge update
- _maybe_reset_judges     — Variant C: periodic judge reinitialization
- _build_meta_holdout     — Variant E: construct the fixed held-out pair set
- _maybe_meta_update_judges — Variant E: corrective step when accuracy drifts
- Variant A (asymmetric update rate) is a single `continue` guard in the inner
  loop and does not benefit from extraction.

All four are controlled by TrainingConfig flags so they compose freely: a run
can combine any subset of A/B/C/D by setting the corresponding config fields.

Judge pretraining (MSE regression on q* until Spearman >= target) also lives
here. The pretrained checkpoint is saved once per seed so that all variant runs
in M5 can share the same starting point.
"""

import logging
import pathlib
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import spearmanr

from coevolution.config import TrainingConfig, WorldConfig
from coevolution.diagnostics import compute_diagnostics
from coevolution.judge import Judge
from coevolution.policy import Policy
from coevolution.utils import save_jsonl
from coevolution.world import all_outputs, q_star_table

logger = logging.getLogger(__name__)

# Type alias for the JIT-compiled step functions returned by the factories below.
_StepFn = Callable[
    [dict, optax.OptState, jnp.ndarray, jnp.ndarray],
    tuple[dict, optax.OptState, jnp.ndarray],
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _binary_cross_entropy(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    """Numerically stable binary cross-entropy from logits.

    Uses the log-sum-exp identity to avoid overflow/underflow:
        BCE(logit, label) = max(logit, 0) - logit*label + log(1 + exp(-|logit|))

    Args:
        logits: Unbounded real values of shape (...).
        labels: Binary targets in {0, 1} of shape (...).

    Returns:
        Element-wise loss values of shape (...).
    """
    return (
        jnp.maximum(logits, 0)
        - logits * labels
        + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1-D arrays.

    Args:
        x: 1-D NumPy array.
        y: 1-D NumPy array of the same length.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    # Index [0] is the correlation coefficient; [1] is the p-value.
    return float(spearmanr(x, y)[0])


# ---------------------------------------------------------------------------
# JIT-compiled step factory functions
# ---------------------------------------------------------------------------


def make_pretrain_step(
    judge: Judge,
    optimizer: optax.GradientTransformation,
) -> Callable[
    [dict, optax.OptState, jnp.ndarray, jnp.ndarray],
    tuple[dict, optax.OptState, jnp.ndarray],
]:
    """Return a JIT-compiled MSE regression step for judge pretraining.

    During pretraining the judge is shown q* values directly — MSE against
    those targets is the loss. This is the only phase where ground-truth
    quality is visible to the judge.

    Args:
        judge: Judge module instance.
        optimizer: Optax optimiser (typically Adam).

    Returns:
        A compiled function step(params, opt_state, sequences, targets)
        that returns (new_params, new_opt_state, scalar_loss).
    """

    @jax.jit
    def step(
        params: dict,
        opt_state: optax.OptState,
        sequences: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState, jnp.ndarray]:
        def loss_fn(p: dict) -> jnp.ndarray:
            scores = judge.apply(p, sequences)
            return jnp.mean((scores - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return step


def make_policy_train_step(
    policy: Policy,
    cfg: TrainingConfig,
    optimizer: optax.GradientTransformation,
) -> Callable[
    [dict, optax.OptState, jnp.ndarray, jnp.ndarray],
    tuple[dict, optax.OptState, jnp.ndarray],
]:
    """Return a JIT-compiled REINFORCE policy update step.

    Rewards are passed in pre-computed rather than computed inside the step.
    This decouples the gradient computation from the judge call, making the
    step function identical for single-judge and ensemble-averaged rewards.

    The loss is:
        L = -E[R_centred * log π(x)] + entropy_bonus * E[log π(x)]

    where R_centred = R - mean(R) is a variance-reduction baseline.
    The second term minimises E[log π] which is equivalent to maximising
    policy entropy.

    Args:
        policy: Policy module instance.
        cfg: Training configuration (entropy_bonus).
        optimizer: Optax optimiser.

    Returns:
        A compiled function step(policy_params, opt_state, samples, rewards)
        that returns (new_params, new_opt_state, scalar_loss).
    """

    @jax.jit
    def step(
        policy_params: dict,
        opt_state: optax.OptState,
        samples: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> tuple[dict, optax.OptState, jnp.ndarray]:
        def loss_fn(pp: dict) -> jnp.ndarray:
            log_probs = policy.log_prob(pp, samples)
            # stop_gradient: rewards are the fixed signal, not differentiated
            rewards_centred = jax.lax.stop_gradient(rewards - rewards.mean())
            reinforce_loss = -jnp.mean(rewards_centred * log_probs)
            # Maximise entropy by penalising low-entropy (high log-prob) distributions
            entropy_loss = jnp.mean(log_probs)
            return reinforce_loss + cfg.entropy_bonus * entropy_loss

        loss, grads = jax.value_and_grad(loss_fn)(policy_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(policy_params, updates)
        return new_params, new_opt_state, loss

    return step


def make_judge_train_step(
    judge: Judge,
    optimizer: optax.GradientTransformation,
) -> Callable[
    [dict, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[dict, optax.OptState, jnp.ndarray],
]:
    """Return a JIT-compiled judge preference update step.

    Labels are passed in pre-computed so the same step function handles
    both self-labelled pairs (from the co-evolution loop) and ground-truth
    anchor pairs (Variant D) without recompilation.

    Args:
        judge: Judge module instance.
        optimizer: Optax optimiser.

    Returns:
        A compiled function step(judge_params, opt_state, x_a, x_b, labels)
        that returns (new_params, new_opt_state, scalar_loss).
    """

    @jax.jit
    def step(
        judge_params: dict,
        opt_state: optax.OptState,
        x_a: jnp.ndarray,
        x_b: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> tuple[dict, optax.OptState, jnp.ndarray]:
        def loss_fn(jp: dict) -> jnp.ndarray:
            score_a = judge.apply(jp, x_a)
            score_b = judge.apply(jp, x_b)
            logits = score_a - score_b
            return jnp.mean(_binary_cross_entropy(logits, labels))

        loss, grads = jax.value_and_grad(loss_fn)(judge_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(judge_params, updates)
        return new_params, new_opt_state, loss

    return step


# ---------------------------------------------------------------------------
# Variant helpers — each encapsulates exactly one architectural choice
# ---------------------------------------------------------------------------


def _compute_rewards(
    judges_params: list[dict],
    judge: Judge,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    """Return policy rewards, averaged across ensemble members (Variant B).

    For n_judges=1 (baseline) this reduces to a single judge.apply call.
    For n_judges>1 each ensemble member scores independently; the mean is
    the reward signal passed to the policy, which smooths out individual
    judge drift.

    Args:
        judges_params: List of parameter dicts, one per ensemble member.
        judge: Shared Judge module instance (all members use the same arch).
        samples: Policy samples of shape (batch, seq_len).

    Returns:
        Reward vector of shape (batch,).
    """
    return jnp.stack(
        [judge.apply(p, samples) for p in judges_params]
    ).mean(axis=0)


def _update_single_judge(
    j_params: dict,
    j_opt: optax.OptState,
    judge_step: _StepFn,
    judge: Judge,
    x_a: jnp.ndarray,
    x_b: jnp.ndarray,
    outputs_tbl: jnp.ndarray,
    q_star_tbl: jnp.ndarray,
    cfg: TrainingConfig,
    rng: jax.Array,
) -> tuple[dict, optax.OptState, jax.Array]:
    """Update one judge on self-labelled pairs, with optional anchor step (Variant D).

    The self-labelled step is always performed: the judge compares x_a and x_b
    under its current parameters and trains on its own preference label.

    The anchor step (Variant D) is performed only when cfg.anchor_fraction > 0.
    It draws a small batch uniformly from the full output space, labels them
    with ground-truth q* comparisons, and runs an additional judge update.
    This prevents the judge from drifting too far from q* as the policy shifts.

    Args:
        j_params: Current judge parameter dict.
        j_opt: Current judge optimiser state.
        judge_step: JIT-compiled judge update function from make_judge_train_step.
        judge: Judge module instance.
        x_a: First set of policy samples, shape (batch, seq_len).
        x_b: Second set of policy samples, shape (batch, seq_len).
        outputs_tbl: All output sequences, shape (output_space, seq_len).
        q_star_tbl: Ground-truth quality for all outputs, shape (output_space,).
        cfg: Training config (anchor_fraction, batch_size).
        rng: JAX PRNG key; consumed and returned updated when anchor is used.

    Returns:
        Tuple of (new_j_params, new_j_opt, new_rng).
    """
    score_a = jax.lax.stop_gradient(judge.apply(j_params, x_a))
    score_b = jax.lax.stop_gradient(judge.apply(j_params, x_b))
    self_labels = (score_a > score_b).astype(jnp.float32)
    j_params, j_opt, _ = judge_step(j_params, j_opt, x_a, x_b, self_labels)

    if cfg.anchor_fraction > 0.0:
        n_anchor = max(1, int(cfg.batch_size * cfg.anchor_fraction))
        rng, rng_anc_a, rng_anc_b = jax.random.split(rng, 3)
        idx_a = jax.random.randint(rng_anc_a, (n_anchor,), 0, len(outputs_tbl))
        idx_b = jax.random.randint(rng_anc_b, (n_anchor,), 0, len(outputs_tbl))
        anc_labels = (q_star_tbl[idx_a] > q_star_tbl[idx_b]).astype(jnp.float32)
        j_params, j_opt, _ = judge_step(
            j_params, j_opt,
            outputs_tbl[idx_a], outputs_tbl[idx_b], anc_labels,
        )

    return j_params, j_opt, rng


def _maybe_reset_judges(
    t: int,
    judges_params: list[dict],
    judges_opt_states: list[optax.OptState],
    reset_judges_params: list[dict],
    judge_optimizer: optax.GradientTransformation,
    cfg: TrainingConfig,
) -> tuple[list[dict], list[optax.OptState]]:
    """Reset judges to their pretrained state on schedule (Variant C).

    When cfg.reset_every <= 0 this is a no-op, so the baseline and all other
    variants pass through unchanged.

    Args:
        t: Current outer iteration index (0-based).
        judges_params: Current judge parameter dicts.
        judges_opt_states: Current judge optimiser states.
        reset_judges_params: Immutable pretrained parameter dicts to restore.
        judge_optimizer: Judge optimiser (used to reinitialise opt state).
        cfg: Training config (reset_every).

    Returns:
        Tuple of (judges_params, judges_opt_states), reset if scheduled.
    """
    if cfg.reset_every > 0 and (t + 1) % cfg.reset_every == 0:
        logger.info("Resetting judges to pretrained state at iteration %d", t + 1)
        judges_params = list(reset_judges_params)
        judges_opt_states = [judge_optimizer.init(p) for p in judges_params]
    return judges_params, judges_opt_states


def _build_meta_holdout(
    outputs_tbl: jnp.ndarray,
    q_star_tbl: jnp.ndarray,
    cfg: TrainingConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None:
    """Build the fixed held-out pair set for the meta-judge (Variant E).

    Uses a deterministic key so the same pairs are used across all seeds and
    variant runs, making held-out accuracy comparable across experiments.

    Returns None when meta_holdout_fraction == 0.0.

    Returns:
        Tuple of (holdout_xa, holdout_xb, holdout_labels) where labels are
        float32 indicators: 1.0 if q*(x_a) > q*(x_b), else 0.0.
    """
    if cfg.meta_holdout_fraction <= 0.0:
        return None

    n = max(1, int(len(outputs_tbl) * cfg.meta_holdout_fraction))
    # Fixed key so the held-out set is identical across all seeds and variants.
    key = jax.random.PRNGKey(0)
    key_a, key_b = jax.random.split(key)
    idx_a = jax.random.randint(key_a, (n,), 0, len(outputs_tbl))
    idx_b = jax.random.randint(key_b, (n,), 0, len(outputs_tbl))
    labels = (q_star_tbl[idx_a] > q_star_tbl[idx_b]).astype(jnp.float32)
    return outputs_tbl[idx_a], outputs_tbl[idx_b], labels


def _maybe_meta_update_judges(
    judges_params: list[dict],
    judges_opt_states: list[optax.OptState],
    judge_step: _StepFn,
    judge: Judge,
    meta_holdout: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    cfg: TrainingConfig,
) -> tuple[list[dict], list[optax.OptState], float | None]:
    """Run a corrective judge update when held-out accuracy drops (Variant E).

    Computes pairwise accuracy on the fixed held-out set. If it falls below
    cfg.meta_accuracy_threshold, runs one gradient step on the held-out pairs
    using ground-truth q* labels. This is a no-op when meta_holdout is None.

    Returns:
        Tuple of (judges_params, judges_opt_states, accuracy). Accuracy is None
        when the meta-judge is disabled.
    """
    if meta_holdout is None:
        return judges_params, judges_opt_states, None

    xa, xb, labels = meta_holdout
    updated_params = []
    updated_opt_states = []
    accuracies = []

    for j_params, j_opt in zip(judges_params, judges_opt_states):
        score_a = judge.apply(j_params, xa)
        score_b = judge.apply(j_params, xb)
        accuracy = float(jnp.mean((score_a > score_b) == labels.astype(bool)))
        accuracies.append(accuracy)

        if accuracy < cfg.meta_accuracy_threshold:
            j_params, j_opt, _ = judge_step(j_params, j_opt, xa, xb, labels)

        updated_params.append(j_params)
        updated_opt_states.append(j_opt)

    mean_accuracy = float(np.mean(accuracies))
    return updated_params, updated_opt_states, mean_accuracy


# ---------------------------------------------------------------------------
# High-level training functions
# ---------------------------------------------------------------------------


def pretrain_judge(
    judge: Judge,
    initial_params: dict,
    q_star_tbl: jnp.ndarray,
    outputs_tbl: jnp.ndarray,
    cfg: TrainingConfig,
    rng: jax.Array,
) -> dict:
    """Pretrain the judge via MSE regression on q* until Spearman >= target.

    The judge is shown ground-truth quality values directly but with small
    Gaussian label noise. The noise and early stopping together ensure the
    judge reaches partial — not perfect — alignment, which is the regime
    where co-evolution dynamics are interesting.

    After pretraining ends, the q* labels are withheld permanently. From
    that point the judge trains only on self-labelled pairs from the policy.

    Args:
        judge: Judge module instance.
        initial_params: Randomly initialised Flax parameter dict.
        q_star_tbl: Precomputed q*(x) for all outputs, shape (output_space,).
        outputs_tbl: All output sequences, shape (output_space, seq_len).
        cfg: Training config (lr_judge, batch_size, judge_init_alignment).
        rng: JAX PRNG key; split internally.

    Returns:
        Trained Flax parameter dict achieving Spearman >= cfg.judge_init_alignment.

    Raises:
        RuntimeError: If the target alignment is not reached within max_steps,
            indicating the learning rate or architecture needs adjustment.
    """
    optimizer = optax.adam(cfg.lr_judge)
    opt_state = optimizer.init(initial_params)
    params = initial_params
    pretrain_step = make_pretrain_step(judge, optimizer)

    label_noise_std = 0.1
    check_every = 50
    max_steps = 5_000

    for step_idx in range(max_steps):
        rng, rng_sample, rng_noise = jax.random.split(rng, 3)
        indices = jax.random.randint(
            rng_sample, (cfg.batch_size,), 0, len(outputs_tbl)
        )
        noisy_targets = (
            q_star_tbl[indices]
            + jax.random.normal(rng_noise, (cfg.batch_size,)) * label_noise_std
        )
        params, opt_state, _ = pretrain_step(
            params, opt_state, outputs_tbl[indices], noisy_targets
        )

        if (step_idx + 1) % check_every == 0:
            all_scores = np.array(judge.apply(params, outputs_tbl))
            corr = _spearman(all_scores, np.array(q_star_tbl))
            logger.debug("Pretrain step %d: Spearman=%.3f", step_idx + 1, corr)
            if corr >= cfg.judge_init_alignment:
                logger.info(
                    "Judge pretrained in %d steps: Spearman=%.3f",
                    step_idx + 1,
                    corr,
                )
                return params

    final_corr = _spearman(
        np.array(judge.apply(params, outputs_tbl)), np.array(q_star_tbl)
    )
    raise RuntimeError(
        f"Judge pretraining did not reach target "
        f"Spearman={cfg.judge_init_alignment:.2f} within {max_steps} steps. "
        f"Final Spearman={final_corr:.3f}. "
        "Consider increasing lr_judge or reducing judge_init_alignment."
    )


def run_coevolution(
    policy: Policy,
    judge: Judge,
    world_cfg: WorldConfig,
    cfg: TrainingConfig,
    pretrained_judges_params: list[dict],
    rng: jax.Array,
    results_dir: pathlib.Path | None = None,
) -> list[dict]:
    """Run the full co-evolution experiment and return per-iteration diagnostics.

    Accepts pretrained judge parameters rather than running pretraining
    internally. This is the correct design for variant comparison: all
    variants within a seed share the same pretrained starting point, so
    differences in outcome are attributable to the architectural choice
    rather than to judge initialisation variance.

    The caller is responsible for running pretrain_judge once per seed and
    passing the result here. See scripts/run_baseline.py for the intended
    usage pattern.

    Args:
        policy: Policy module instance.
        judge: Judge module instance.
        world_cfg: World configuration.
        cfg: Training configuration including all variant flags.
        pretrained_judges_params: List of pretrained Flax parameter dicts,
            one per ensemble member (length 1 for the baseline). Produced by
            calling pretrain_judge before this function.
        rng: JAX PRNG key; split internally across all random operations.
            Pass the same key for all variant calls within a seed to ensure
            the policy is initialised identically across variants.
        results_dir: If provided, JSONL diagnostics are written here.

    Returns:
        List of diagnostic dicts, one per outer iteration. Each dict contains
        at minimum {"iteration": int, "seed": int}; M4 extends this with the
        four headline metrics.

    Raises:
        ValueError: If len(pretrained_judges_params) != cfg.n_judges.
    """
    if len(pretrained_judges_params) != cfg.n_judges:
        raise ValueError(
            f"Expected {cfg.n_judges} pretrained judge param dicts "
            f"(cfg.n_judges={cfg.n_judges}), "
            f"got {len(pretrained_judges_params)}."
        )

    # ---- initialise policy ----
    rng, rng_policy = jax.random.split(rng)
    dummy = jnp.zeros((1, world_cfg.seq_len), dtype=jnp.int32)
    policy_params = policy.init(rng_policy, dummy)

    # ---- precompute world tables (used throughout, never shown to judge) ----
    outputs_tbl = all_outputs(world_cfg)
    q_star_tbl = q_star_table(
        world_cfg, jax.random.PRNGKey(world_cfg.seed_for_q_star)
    )

    # Start from the pretrained state; keep an immutable copy for Variant C resets
    judges_params: list[dict] = list(pretrained_judges_params)
    # Immutable reference for periodic resets (Variant C)
    reset_judges_params: list[dict] = list(pretrained_judges_params)

    # Variant E: build fixed held-out set once (None when disabled)
    meta_holdout = _build_meta_holdout(outputs_tbl, q_star_tbl, cfg)

    # ---- optimisers ----
    policy_optimizer = optax.adam(cfg.lr_policy)
    judge_optimizer = optax.adam(cfg.lr_judge)
    policy_opt_state = policy_optimizer.init(policy_params)
    judges_opt_states: list[optax.OptState] = [
        judge_optimizer.init(p) for p in judges_params
    ]

    # ---- JIT-compiled step functions (compiled once, reused every step) ----
    policy_step = make_policy_train_step(policy, cfg, policy_optimizer)
    judge_step = make_judge_train_step(judge, judge_optimizer)

    history: list[dict] = []

    for t in range(cfg.n_iterations):
        logger.info("Iteration %d / %d", t + 1, cfg.n_iterations)

        for inner in range(cfg.steps_per_iter):
            rng, rng_sample = jax.random.split(rng)
            samples = policy.sample(policy_params, rng_sample, cfg.batch_size)

            rewards = _compute_rewards(judges_params, judge, samples)
            policy_params, policy_opt_state, _ = policy_step(
                policy_params, policy_opt_state, samples, rewards
            )

            # Variant A: update judge only every judge_update_every policy steps
            if (inner + 1) % cfg.judge_update_every != 0:
                continue

            rng, rng_a, rng_b = jax.random.split(rng, 3)
            x_a = policy.sample(policy_params, rng_a, cfg.batch_size)
            x_b = policy.sample(policy_params, rng_b, cfg.batch_size)

            for j_idx in range(cfg.n_judges):
                judges_params[j_idx], judges_opt_states[j_idx], rng = (
                    _update_single_judge(
                        judges_params[j_idx],
                        judges_opt_states[j_idx],
                        judge_step,
                        judge,
                        x_a,
                        x_b,
                        outputs_tbl,
                        q_star_tbl,
                        cfg,
                        rng,
                    )
                )

        judges_params, judges_opt_states = _maybe_reset_judges(
            t, judges_params, judges_opt_states,
            reset_judges_params, judge_optimizer, cfg,
        )

        judges_params, judges_opt_states, meta_acc = _maybe_meta_update_judges(
            judges_params, judges_opt_states,
            judge_step, judge, meta_holdout, cfg,
        )

        diag = compute_diagnostics(
            policy_params=policy_params,
            judges_params=judges_params,
            q_star_tbl=q_star_tbl,
            outputs_tbl=outputs_tbl,
            policy=policy,
            judge=judge,
            iteration=t,
            seed=cfg.seed,
        )
        if meta_acc is not None:
            diag["meta_judge_accuracy"] = meta_acc
        history.append(diag)
        logger.info("Iteration %d: %s", t + 1, diag)

    if results_dir is not None:
        save_jsonl(results_dir / f"seed_{cfg.seed}.jsonl", history)

    return history
