"""
Tests for src/coevolution/train.py (Milestone 3).

Run:
    pytest tests/test_train.py

Note: these tests use a reduced config to stay fast. The full 10-iteration
baseline run (the M3 acceptance criterion for wall-time) is exercised by
scripts/run_baseline.py, not here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from coevolution.config import AgentConfig, TrainingConfig, WorldConfig
from coevolution.judge import Judge
from coevolution.policy import Policy
from coevolution.train import (
    _spearman,
    make_judge_train_step,
    make_policy_train_step,
    make_pretrain_step,
    pretrain_judge,
    run_coevolution,
)
from coevolution.world import all_outputs, q_star_table

RNG = jax.random.PRNGKey(0)


@pytest.fixture
def world() -> WorldConfig:
    """Default world configuration."""
    return WorldConfig()


@pytest.fixture
def agent_cfg() -> AgentConfig:
    """Default agent configuration."""
    return AgentConfig()


@pytest.fixture
def fast_cfg() -> TrainingConfig:
    """Reduced training config that makes tests run in seconds.

    Uses a lower judge_init_alignment target so pretraining converges quickly,
    and minimal iterations/steps so the co-evolution loop doesn't time out.
    """
    return TrainingConfig(
        seed=0,
        n_iterations=2,
        steps_per_iter=5,
        batch_size=16,
        lr_policy=3e-4,
        lr_judge=3e-4,
        entropy_bonus=0.01,
        judge_init_alignment=0.4,  # lower than headline 0.7 for test speed
    )


@pytest.fixture
def initialized_networks(
    world: WorldConfig, agent_cfg: AgentConfig
) -> tuple[Policy, dict, Judge, dict]:
    """Return initialised policy and judge with their parameter dicts."""
    policy = Policy(cfg=agent_cfg, world=world)
    judge = Judge(cfg=agent_cfg, world=world)
    dummy = jnp.zeros((1, world.seq_len), dtype=jnp.int32)
    rng_p, rng_j = jax.random.split(RNG)
    policy_params = policy.init(rng_p, dummy)
    judge_params = judge.init(rng_j, dummy)
    return policy, policy_params, judge, judge_params


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_spearman_perfect_correlation() -> None:
    """_spearman must return 1.0 for identical arrays.

    Why: a broken Spearman implementation would give wrong stopping criteria
    during pretraining, causing the judge to stop too early or not at all.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(_spearman(x, x) - 1.0) < 1e-6


def test_spearman_anti_correlation() -> None:
    """_spearman must return -1.0 for reversed arrays."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(_spearman(x, x[::-1]) + 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Pretrain step
# ---------------------------------------------------------------------------


def test_pretrain_step_reduces_loss(
    initialized_networks: tuple[Policy, dict, Judge, dict],
    world: WorldConfig,
) -> None:
    """A single pretrain step must reduce MSE loss on the same batch.

    Why: if the gradient direction is wrong the judge will never reach
    the target alignment, regardless of how long pretraining runs.
    """
    _, _, judge, judge_params = initialized_networks
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(judge_params)
    step = make_pretrain_step(judge, optimizer)

    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))
    sequences = outputs[:16]
    targets = q_star[:16]

    params_before = judge_params
    params_after, _, loss_after = step(params_before, opt_state, sequences, targets)

    # Loss should be finite after one step
    assert float(jnp.isfinite(loss_after))


def test_pretrain_step_no_nan(
    initialized_networks: tuple[Policy, dict, Judge, dict],
    world: WorldConfig,
) -> None:
    """Pretrain step must not produce NaN parameters.

    Why: NaN params propagate silently through all downstream computations,
    corrupting the entire run without raising an error.
    """
    _, _, judge, judge_params = initialized_networks
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(judge_params)
    step = make_pretrain_step(judge, optimizer)

    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))
    new_params, _, _ = step(judge_params, opt_state, outputs[:16], q_star[:16])

    leaves = jax.tree_util.tree_leaves(new_params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


# ---------------------------------------------------------------------------
# Policy train step
# ---------------------------------------------------------------------------


def test_policy_step_no_nan(
    initialized_networks: tuple[Policy, dict, Judge, dict],
    world: WorldConfig,
    fast_cfg: TrainingConfig,
) -> None:
    """Policy update step must not produce NaN parameters.

    Why: NaN in the policy breaks sampling immediately, crashing the entire
    training loop on the next iteration.
    """
    policy, policy_params, judge, judge_params = initialized_networks
    optimizer = optax.adam(fast_cfg.lr_policy)
    opt_state = optimizer.init(policy_params)
    step = make_policy_train_step(policy, fast_cfg, optimizer)

    rng_s = jax.random.PRNGKey(1)
    samples = policy.sample(policy_params, rng_s, fast_cfg.batch_size)
    rewards = judge.apply(judge_params, samples)

    new_params, _, loss = step(policy_params, opt_state, samples, rewards)

    assert bool(jnp.isfinite(loss))
    leaves = jax.tree_util.tree_leaves(new_params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


# ---------------------------------------------------------------------------
# Judge train step
# ---------------------------------------------------------------------------


def test_judge_step_no_nan(
    initialized_networks: tuple[Policy, dict, Judge, dict],
    world: WorldConfig,
    fast_cfg: TrainingConfig,
) -> None:
    """Judge preference update step must not produce NaN parameters.

    Why: NaN in the judge collapses all rewards to NaN, stopping policy
    learning entirely.
    """
    policy, policy_params, judge, judge_params = initialized_networks
    optimizer = optax.adam(fast_cfg.lr_judge)
    opt_state = optimizer.init(judge_params)
    step = make_judge_train_step(judge, optimizer)

    rng_a, rng_b = jax.random.split(jax.random.PRNGKey(2))
    x_a = policy.sample(policy_params, rng_a, fast_cfg.batch_size)
    x_b = policy.sample(policy_params, rng_b, fast_cfg.batch_size)
    score_a = jax.lax.stop_gradient(judge.apply(judge_params, x_a))
    score_b = jax.lax.stop_gradient(judge.apply(judge_params, x_b))
    labels = (score_a > score_b).astype(jnp.float32)

    new_params, _, loss = step(judge_params, opt_state, x_a, x_b, labels)

    assert bool(jnp.isfinite(loss))
    leaves = jax.tree_util.tree_leaves(new_params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


# ---------------------------------------------------------------------------
# pretrain_judge
# ---------------------------------------------------------------------------


def test_pretrain_judge_reaches_target(
    initialized_networks: tuple[Policy, dict, Judge, dict],
    world: WorldConfig,
    fast_cfg: TrainingConfig,
) -> None:
    """pretrain_judge must reach the target Spearman correlation.

    Why: if pretraining fails, the judge starts the co-evolution loop
    randomly aligned with q*, making every subsequent run incomparable.
    """
    _, _, judge, judge_params = initialized_networks
    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))

    trained_params = pretrain_judge(
        judge, judge_params, q_star, outputs, fast_cfg, RNG
    )

    all_scores = np.array(judge.apply(trained_params, outputs))
    corr = _spearman(all_scores, np.array(q_star))
    assert corr >= fast_cfg.judge_init_alignment


# ---------------------------------------------------------------------------
# run_coevolution
# ---------------------------------------------------------------------------


@pytest.fixture
def pretrained_judge_params(
    world: WorldConfig, agent_cfg: AgentConfig, fast_cfg: TrainingConfig
) -> list[dict]:
    """Pretrain a single judge to fast_cfg.judge_init_alignment.

    Shared across run_coevolution tests so pretraining only happens once,
    matching the intended usage pattern in the scripts.
    """
    judge = Judge(cfg=agent_cfg, world=world)
    dummy = jnp.zeros((1, world.seq_len), dtype=jnp.int32)
    initial_params = judge.init(RNG, dummy)
    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))
    trained = pretrain_judge(judge, initial_params, q_star, outputs, fast_cfg, RNG)
    return [trained]


def test_run_coevolution_returns_correct_length(
    world: WorldConfig,
    agent_cfg: AgentConfig,
    fast_cfg: TrainingConfig,
    pretrained_judge_params: list[dict],
) -> None:
    """run_coevolution must return exactly n_iterations diagnostic records.

    Why: a missing or duplicated iteration record would misalign the x-axis
    of the headline plot.
    """
    policy = Policy(cfg=agent_cfg, world=world)
    judge = Judge(cfg=agent_cfg, world=world)
    history = run_coevolution(
        policy, judge, world, fast_cfg, pretrained_judge_params, RNG
    )
    assert len(history) == fast_cfg.n_iterations


def test_run_coevolution_records_have_iteration_and_seed(
    world: WorldConfig,
    agent_cfg: AgentConfig,
    fast_cfg: TrainingConfig,
    pretrained_judge_params: list[dict],
) -> None:
    """Each diagnostic record must contain iteration and seed fields.

    Why: iteration and seed are the primary keys used to join records across
    runs when building the headline plot.
    """
    policy = Policy(cfg=agent_cfg, world=world)
    judge = Judge(cfg=agent_cfg, world=world)
    history = run_coevolution(
        policy, judge, world, fast_cfg, pretrained_judge_params, RNG
    )
    for i, record in enumerate(history):
        assert record["iteration"] == i
        assert record["seed"] == fast_cfg.seed


def test_run_coevolution_wrong_judge_count_raises(
    world: WorldConfig,
    agent_cfg: AgentConfig,
    fast_cfg: TrainingConfig,
    pretrained_judge_params: list[dict],
) -> None:
    """run_coevolution must raise ValueError if judge count mismatches cfg.n_judges.

    Why: a silent mismatch here would cause ensemble averaging over the wrong
    number of judges without any error signal.
    """
    policy = Policy(cfg=agent_cfg, world=world)
    judge = Judge(cfg=agent_cfg, world=world)
    wrong_params = pretrained_judge_params + pretrained_judge_params  # 2 instead of 1
    with pytest.raises(ValueError, match="n_judges"):
        run_coevolution(
            policy, judge, world, fast_cfg, wrong_params, RNG
        )
