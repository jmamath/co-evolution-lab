"""
Tests for src/coevolution/diagnostics.py (Milestone 4).

Run:
    pytest tests/test_diagnostics.py

Each metric is tested against three canonical cases whose expected values are
derivable analytically: a uniform (identity) policy, a peaked (collapsed)
policy, and a judge whose scores are perfectly aligned or anti-aligned with q*.
This makes the tests self-documenting and independent of the network training
dynamics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coevolution.config import AgentConfig, WorldConfig
from coevolution.diagnostics import (
    compute_diagnostics,
    detect_ceiling,
    judge_alignment,
    judge_perceived_quality,
    policy_entropy,
    policy_quality,
)
from coevolution.judge import Judge
from coevolution.policy import Policy
from coevolution.world import all_outputs, q_star_table

RNG = jax.random.PRNGKey(0)
N = 4096  # full output space size for default WorldConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def world() -> WorldConfig:
    return WorldConfig()


@pytest.fixture
def q_star(world: WorldConfig) -> np.ndarray:
    tbl = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))
    return np.array(tbl)


@pytest.fixture
def uniform_probs() -> np.ndarray:
    """Uniform distribution over all N outputs — the identity/baseline policy."""
    return np.full(N, 1.0 / N)


@pytest.fixture
def peaked_probs() -> np.ndarray:
    """All probability mass on the first output — extreme distributional collapse."""
    p = np.zeros(N)
    p[0] = 1.0
    return p


# ---------------------------------------------------------------------------
# policy_quality
# ---------------------------------------------------------------------------


def test_policy_quality_uniform(
    uniform_probs: np.ndarray, q_star: np.ndarray
) -> None:
    """Uniform policy must give E[q*] = mean(q*).

    > **Why:** if this fails, the expectation formula is wrong and every
    quality reading in the headline plot will be biased.
    """
    expected = float(np.mean(q_star))
    result = policy_quality(jnp.array(uniform_probs), jnp.array(q_star))
    assert abs(result - expected) < 1e-4


def test_policy_quality_peaked(
    peaked_probs: np.ndarray, q_star: np.ndarray
) -> None:
    """Fully peaked policy must give the quality of the single selected sequence.

    > **Why:** ensures that a collapsed policy is correctly diagnosed as
    optimising exactly one sequence, not averaging over the space.
    """
    expected = float(q_star[0])
    result = policy_quality(jnp.array(peaked_probs), jnp.array(q_star))
    assert abs(result - expected) < 1e-4


# ---------------------------------------------------------------------------
# policy_entropy
# ---------------------------------------------------------------------------


def test_policy_entropy_uniform(uniform_probs: np.ndarray) -> None:
    """Uniform distribution must have maximum entropy = log(N).

    > **Why:** if entropy reads low for a uniform policy the collapse
    diagnostic will fire falsely, making every run look like a failure.
    """
    expected = float(np.log(N))
    result = policy_entropy(jnp.array(uniform_probs))
    assert abs(result - expected) < 1e-3


def test_policy_entropy_peaked(peaked_probs: np.ndarray) -> None:
    """Fully peaked policy must have entropy = 0.

    > **Why:** entropy = 0 is the definition of collapse; anything above zero
    for a point mass indicates a bug in the log-zero handling.
    """
    result = policy_entropy(jnp.array(peaked_probs))
    assert abs(result) < 1e-4


def test_policy_entropy_non_negative(uniform_probs: np.ndarray) -> None:
    """Entropy must never be negative regardless of the input distribution.

    > **Why:** negative entropy would silently invert the collapse narrative
    (lower entropy would look like progress rather than failure).
    """
    result = policy_entropy(jnp.array(uniform_probs))
    assert result >= 0.0


# ---------------------------------------------------------------------------
# judge_alignment
# ---------------------------------------------------------------------------


def test_judge_alignment_perfect(q_star: np.ndarray) -> None:
    """Judge scores equal to q* must give Spearman = 1.0.

    > **Why:** a pretrained judge should start near this value; if the metric
    reads below 1.0 for an exact copy of q* the alignment trend will be
    systematically underestimated.
    """
    result = judge_alignment(jnp.array(q_star), jnp.array(q_star))
    assert abs(result - 1.0) < 1e-6


def test_judge_alignment_inverted(q_star: np.ndarray) -> None:
    """Inverted judge scores must give Spearman = -1.0.

    > **Why:** confirms the sign convention is correct — a judge that always
    prefers worse sequences should be detectable as negatively aligned.
    """
    result = judge_alignment(jnp.array(-q_star), jnp.array(q_star))
    assert abs(result + 1.0) < 1e-6


def test_judge_alignment_random(q_star: np.ndarray) -> None:
    """Random judge scores must give Spearman correlation near zero.

    > **Why:** a randomly initialised judge should have no alignment signal;
    any strong non-zero reading here would indicate the metric is picking up
    spurious structure in the score initialisation.
    """
    rng = np.random.default_rng(42)
    random_scores = rng.standard_normal(N)
    result = judge_alignment(jnp.array(random_scores), jnp.array(q_star))
    assert abs(result) < 0.1


# ---------------------------------------------------------------------------
# judge_perceived_quality
# ---------------------------------------------------------------------------


def test_judge_perceived_quality_uniform(
    uniform_probs: np.ndarray, q_star: np.ndarray
) -> None:
    """Uniform policy + perfect judge must give perceived quality = mean(q*).

    > **Why:** when the judge is perfect, perceived quality and true quality
    must agree for a uniform policy; any divergence signals a formula error.
    """
    expected = float(np.mean(q_star))
    result = judge_perceived_quality(
        jnp.array(uniform_probs), jnp.array(q_star)
    )
    assert abs(result - expected) < 1e-4


def test_judge_perceived_quality_peaked(
    peaked_probs: np.ndarray,
) -> None:
    """Peaked policy + arbitrary judge must give score of the single selected sequence.

    > **Why:** confirms the expectation collapses to a point mass correctly,
    which is the degenerate-collapse case the experiment tries to avoid.
    """
    scores = np.arange(N, dtype=np.float32)
    expected = float(scores[0])
    result = judge_perceived_quality(jnp.array(peaked_probs), jnp.array(scores))
    assert abs(result - expected) < 1e-4


# ---------------------------------------------------------------------------
# detect_ceiling
# ---------------------------------------------------------------------------


def test_detect_ceiling_peak_in_middle() -> None:
    """detect_ceiling must locate the iteration with the highest quality.

    > **Why:** a wrong peak index would report the ceiling effect as occurring
    at the wrong stage of training, invalidating the writeup's main claim.
    """
    history = [
        {"policy_quality": 0.1},
        {"policy_quality": 0.9},
        {"policy_quality": 0.3},
    ]
    result = detect_ceiling(history)
    assert result["iteration_at_peak"] == 1
    assert abs(result["peak_quality"] - 0.9) < 1e-6
    assert abs(result["final_quality"] - 0.3) < 1e-6


def test_detect_ceiling_degradation_positive_when_quality_falls() -> None:
    """degradation_pct must be positive when final quality is below peak.

    > **Why:** a negative or zero degradation for a clearly falling curve
    would misreport the ceiling finding as "no degradation observed."
    """
    history = [
        {"policy_quality": 1.0},
        {"policy_quality": 2.0},
        {"policy_quality": 1.0},
    ]
    result = detect_ceiling(history)
    assert result["degradation_pct"] > 0.0


def test_detect_ceiling_no_degradation_when_monotone() -> None:
    """degradation_pct must be zero (or negative) when quality is monotone increasing.

    > **Why:** a monotone run has no ceiling; reporting any positive degradation
    here would be a false positive that overstates the severity of the effect.
    """
    history = [
        {"policy_quality": 0.1},
        {"policy_quality": 0.5},
        {"policy_quality": 0.9},
    ]
    result = detect_ceiling(history)
    assert result["degradation_pct"] <= 0.0


def test_detect_ceiling_empty_raises() -> None:
    """detect_ceiling must raise ValueError on an empty history list.

    > **Why:** silently returning a default would mask the bug where
    run_coevolution produced no output, hiding the real failure.
    """
    with pytest.raises(ValueError):
        detect_ceiling([])


def test_detect_ceiling_missing_key_raises() -> None:
    """detect_ceiling must raise ValueError when policy_quality key is absent.

    > **Why:** this would happen if compute_diagnostics is called before M4
    is wired in; a clear error is better than an AttributeError deep in numpy.
    """
    with pytest.raises(ValueError, match="policy_quality"):
        detect_ceiling([{"iteration": 0, "seed": 0}])


# ---------------------------------------------------------------------------
# compute_diagnostics integration
# ---------------------------------------------------------------------------


@pytest.fixture
def network_fixtures(world: WorldConfig) -> tuple[Policy, dict, Judge, dict]:
    agent_cfg = AgentConfig()
    policy = Policy(cfg=agent_cfg, world=world)
    judge = Judge(cfg=agent_cfg, world=world)
    dummy = jnp.zeros((1, world.seq_len), dtype=jnp.int32)
    rng_p, rng_j = jax.random.split(RNG)
    return policy, policy.init(rng_p, dummy), judge, judge.init(rng_j, dummy)


def test_compute_diagnostics_keys(
    world: WorldConfig,
    network_fixtures: tuple[Policy, dict, Judge, dict],
) -> None:
    """compute_diagnostics must return all four headline metric keys.

    > **Why:** a missing key would cause a KeyError when building the headline
    plot, failing silently until the plotting script runs.
    """
    policy, policy_params, judge, judge_params = network_fixtures
    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))

    diag = compute_diagnostics(
        policy_params=policy_params,
        judges_params=[judge_params],
        q_star_tbl=q_star,
        outputs_tbl=outputs,
        policy=policy,
        judge=judge,
        iteration=0,
        seed=0,
    )

    for key in (
        "iteration",
        "seed",
        "policy_quality",
        "judge_alignment",
        "policy_entropy",
        "judge_perceived_quality",
    ):
        assert key in diag, f"Missing key: {key}"


def test_compute_diagnostics_values_finite(
    world: WorldConfig,
    network_fixtures: tuple[Policy, dict, Judge, dict],
) -> None:
    """All metric values returned by compute_diagnostics must be finite.

    > **Why:** a NaN or Inf in any metric would silently corrupt the JSONL
    output and produce broken plots without raising an error.
    """
    policy, policy_params, judge, judge_params = network_fixtures
    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))

    diag = compute_diagnostics(
        policy_params=policy_params,
        judges_params=[judge_params],
        q_star_tbl=q_star,
        outputs_tbl=outputs,
        policy=policy,
        judge=judge,
        iteration=3,
        seed=1,
    )

    for key in (
        "policy_quality",
        "judge_alignment",
        "policy_entropy",
        "judge_perceived_quality",
    ):
        assert np.isfinite(diag[key]), f"{key} = {diag[key]} is not finite"


def test_compute_diagnostics_entropy_in_valid_range(
    world: WorldConfig,
    network_fixtures: tuple[Policy, dict, Judge, dict],
) -> None:
    """policy_entropy must lie in [0, log(output_space_size)].

    > **Why:** entropy outside this range is mathematically impossible and
    would indicate a normalisation error in the policy probability computation.
    """
    policy, policy_params, judge, judge_params = network_fixtures
    outputs = all_outputs(world)
    q_star = q_star_table(world, jax.random.PRNGKey(world.seed_for_q_star))
    output_space = world.vocab_size ** world.seq_len

    diag = compute_diagnostics(
        policy_params=policy_params,
        judges_params=[judge_params],
        q_star_tbl=q_star,
        outputs_tbl=outputs,
        policy=policy,
        judge=judge,
        iteration=0,
        seed=0,
    )

    assert diag["policy_entropy"] >= 0.0
    assert diag["policy_entropy"] <= np.log(output_space) + 1e-3
