"""
Tests for src/coevolution/judge.py (Milestone 2).

Run:
    pytest tests/test_judge.py
"""

import jax
import jax.numpy as jnp
import pytest

from coevolution.config import AgentConfig, WorldConfig
from coevolution.judge import Judge

RNG = jax.random.PRNGKey(0)


@pytest.fixture
def world() -> WorldConfig:
    """Default world configuration."""
    return WorldConfig()


@pytest.fixture
def cfg() -> AgentConfig:
    """Default agent configuration."""
    return AgentConfig()


@pytest.fixture
def judge_with_params(
    world: WorldConfig, cfg: AgentConfig
) -> tuple[Judge, dict]:
    """Initialised judge with a dummy forward pass."""
    judge = Judge(cfg=cfg, world=world)
    dummy = jnp.zeros((1, world.seq_len), dtype=jnp.int32)
    params = judge.init(RNG, dummy)
    return judge, params


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_judge_output_shape(
    judge_with_params: tuple[Judge, dict], world: WorldConfig
) -> None:
    """__call__ must return one scalar score per sequence in the batch.

    Why: the preference label sigmoid(J(x_a) - J(x_b)) requires exactly one
    scalar per sequence; a wrong shape would silently corrupt pair comparisons.
    """
    judge, params = judge_with_params
    B = 8
    x = jnp.zeros((B, world.seq_len), dtype=jnp.int32)
    scores = judge.apply(params, x)
    assert scores.shape == (B,)


def test_judge_scores_are_finite(
    judge_with_params: tuple[Judge, dict], world: WorldConfig
) -> None:
    """Judge scores must be finite for all-zero inputs.

    Why: NaN or Inf scores would propagate into the preference labels and
    corrupt every subsequent policy update silently.
    """
    judge, params = judge_with_params
    x = jnp.zeros((16, world.seq_len), dtype=jnp.int32)
    scores = judge.apply(params, x)
    assert bool(jnp.all(jnp.isfinite(scores)))


def test_judge_scores_vary_across_sequences(
    judge_with_params: tuple[Judge, dict], world: WorldConfig
) -> None:
    """The judge must produce different scores for different sequences.

    Why: a constant judge provides no learning signal — every pair preference
    would be a coin flip regardless of sequence content.
    """
    judge, params = judge_with_params
    rng = jax.random.PRNGKey(1)
    x = jax.random.randint(rng, (64, world.seq_len), 0, world.vocab_size)
    scores = judge.apply(params, x)
    assert float(scores.std()) > 0.0


def test_judge_preference_is_in_zero_one(
    judge_with_params: tuple[Judge, dict], world: WorldConfig
) -> None:
    """sigmoid(J(x_a) - J(x_b)) must lie in (0, 1) for all pairs.

    Why: preference values outside (0, 1) would indicate a numerical issue
    in the scoring chain before any training has occurred.
    """
    judge, params = judge_with_params
    rng_a, rng_b = jax.random.split(jax.random.PRNGKey(2))
    x_a = jax.random.randint(rng_a, (32, world.seq_len), 0, world.vocab_size)
    x_b = jax.random.randint(rng_b, (32, world.seq_len), 0, world.vocab_size)
    scores_a = judge.apply(params, x_a)
    scores_b = judge.apply(params, x_b)
    prefs = jax.nn.sigmoid(scores_a - scores_b)
    assert bool(jnp.all(prefs > 0))
    assert bool(jnp.all(prefs < 1))


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def test_judge_parameter_count(
    judge_with_params: tuple[Judge, dict],
) -> None:
    """Total judge parameter count must stay well under 100K.

    Why: same compute budget rationale as the policy; the two networks
    should be comparable in capacity so neither dominates by default.
    """
    _, params = judge_with_params
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert total < 100_000
