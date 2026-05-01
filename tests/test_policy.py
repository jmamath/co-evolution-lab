"""
Tests for src/coevolution/policy.py (Milestone 2).

Run:
    pytest tests/test_policy.py
"""

import jax
import jax.numpy as jnp
import pytest

from coevolution.config import AgentConfig, WorldConfig
from coevolution.policy import Policy
from coevolution.world import all_outputs

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
def policy_with_params(
    world: WorldConfig, cfg: AgentConfig
) -> tuple[Policy, dict]:
    """Initialised policy with a dummy forward pass."""
    policy = Policy(cfg=cfg, world=world)
    dummy = jnp.zeros((1, world.seq_len), dtype=jnp.int32)
    params = policy.init(RNG, dummy)
    return policy, params


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_pass_shape(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """__call__ must return logits of shape (batch, seq_len, vocab_size).

    Why: any shape mismatch propagates silently into log_prob and the
    REINFORCE gradient, producing wrong updates without an obvious error.
    """
    policy, params = policy_with_params
    B = 4
    x = jnp.zeros((B, world.seq_len), dtype=jnp.int32)
    logits = policy.apply(params, x)
    assert logits.shape == (B, world.seq_len, world.vocab_size)


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------


def test_log_prob_shape(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """log_prob must return one scalar per sequence in the batch.

    Why: the REINFORCE loss sums log_prob * reward over the batch; a shape
    mismatch here would silently broadcast and corrupt the gradient.
    """
    policy, params = policy_with_params
    B = 8
    x = jnp.zeros((B, world.seq_len), dtype=jnp.int32)
    lp = policy.log_prob(params, x)
    assert lp.shape == (B,)


def test_log_prob_non_positive(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """Log probabilities must be <= 0 for all sequences.

    Why: a positive log-prob implies probability > 1, which indicates a
    numerical error in the softmax or indexing logic.
    """
    policy, params = policy_with_params
    x = jnp.zeros((16, world.seq_len), dtype=jnp.int32)
    lp = policy.log_prob(params, x)
    assert bool(jnp.all(lp <= 0))


def test_log_prob_sums_to_one_over_full_space(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """exp(log_prob) must sum to 1 over the complete output space.

    Why: if the policy is not a valid probability distribution, the
    expected-quality diagnostic E_x~π[q*(x)] is computed incorrectly.
    """
    policy, params = policy_with_params
    outputs = all_outputs(world)  # (4096, seq_len)
    log_probs = policy.log_prob(params, outputs)
    total = float(jnp.exp(log_probs).sum())
    assert abs(total - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_shape(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """sample must return exactly n sequences of length seq_len.

    Why: downstream batching logic assumes a fixed output shape; a mismatch
    would cause silent truncation or padding.
    """
    policy, params = policy_with_params
    n = 32
    samples = policy.sample(params, RNG, n)
    assert samples.shape == (n, world.seq_len)


def test_sample_token_range(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """All sampled tokens must lie within [0, vocab_size).

    Why: out-of-range tokens would corrupt embedding lookups and q*
    evaluations without raising an error in JAX.
    """
    policy, params = policy_with_params
    samples = policy.sample(params, RNG, 64)
    assert int(samples.min()) >= 0
    assert int(samples.max()) < world.vocab_size


# ---------------------------------------------------------------------------
# greedy_decode
# ---------------------------------------------------------------------------


def test_greedy_decode_shape(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """greedy_decode must return a single sequence of length seq_len.

    Why: the decoded sequence is used at analysis time to track the policy
    mode; a wrong shape would break all rendering and diagnostic calls.
    """
    policy, params = policy_with_params
    seq = policy.greedy_decode(params)
    assert seq.shape == (world.seq_len,)


def test_greedy_decode_token_range(
    policy_with_params: tuple[Policy, dict], world: WorldConfig
) -> None:
    """All greedy-decoded tokens must lie within [0, vocab_size)."""
    policy, params = policy_with_params
    seq = policy.greedy_decode(params)
    assert int(seq.min()) >= 0
    assert int(seq.max()) < world.vocab_size


def test_greedy_decode_is_deterministic(
    policy_with_params: tuple[Policy, dict],
) -> None:
    """greedy_decode must return the same sequence on repeated calls.

    Why: it takes no rng; any non-determinism would indicate a hidden
    stateful side-effect in the network.
    """
    policy, params = policy_with_params
    seq1 = policy.greedy_decode(params)
    seq2 = policy.greedy_decode(params)
    assert jnp.array_equal(seq1, seq2)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def test_policy_parameter_count(
    policy_with_params: tuple[Policy, dict],
) -> None:
    """Total policy parameter count must stay well under 100K.

    Why: the experiment is designed for CPU inference; a runaway parameter
    count would violate the compute budget and slow every training run.
    """
    _, params = policy_with_params
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert total < 100_000
