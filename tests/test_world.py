"""
Tests for src/coevolution/world.py (Milestone 1).

Run:
    pytest tests/test_world.py
"""

import jax
import jax.numpy as jnp
import pytest

from coevolution.config import WorldConfig
from coevolution.world import (
    VOCAB_SYMBOLS,
    all_outputs,
    make_q_star,
    q_star_table,
    render_sequence,
)

# Fixed RNG keys used across tests. Two distinct keys let us verify that
# different seeds produce different weights without relying on randomness.
RNG_A = jax.random.PRNGKey(0)
RNG_B = jax.random.PRNGKey(1)


@pytest.fixture
def default_cfg() -> WorldConfig:
    """Return the default WorldConfig used in all headline experiments."""
    return WorldConfig()


# ---------------------------------------------------------------------------
# all_outputs
# ---------------------------------------------------------------------------


def test_all_outputs_shape(default_cfg: WorldConfig) -> None:
    """all_outputs must return exactly vocab_size ** seq_len rows of length seq_len.

    Why: if the output space is incomplete or duplicated, every downstream
    diagnostic is computed over the wrong distribution.
    """
    outputs = all_outputs(default_cfg)
    expected_rows = default_cfg.vocab_size**default_cfg.seq_len
    assert outputs.shape == (expected_rows, default_cfg.seq_len)


def test_all_outputs_token_range(default_cfg: WorldConfig) -> None:
    """Every token in all_outputs must lie within [0, vocab_size).

    Why: out-of-range indices would cause silent index wrapping in JAX and
    corrupt q* evaluations without raising an error.
    """
    outputs = all_outputs(default_cfg)
    assert int(outputs.min()) >= 0
    assert int(outputs.max()) < default_cfg.vocab_size


def test_all_outputs_unique_rows(default_cfg: WorldConfig) -> None:
    """all_outputs must cover every sequence exactly once.

    Why: duplicate rows would cause certain sequences to be over-weighted in
    exact diagnostics and others to be missed entirely.
    """
    outputs = all_outputs(default_cfg)
    # Convert each row to a tuple and count unique entries
    rows_as_tuples = set(tuple(int(t) for t in row) for row in outputs)
    expected_count = default_cfg.vocab_size**default_cfg.seq_len
    assert len(rows_as_tuples) == expected_count


# ---------------------------------------------------------------------------
# make_q_star / q_star_table
# ---------------------------------------------------------------------------


def test_q_star_deterministic(default_cfg: WorldConfig) -> None:
    """q* must return identical scores for the same rng and config.

    Why: the entire experiment depends on a fixed oracle. Any non-determinism
    would make comparisons across runs meaningless.
    """
    outputs = all_outputs(default_cfg)
    q1 = make_q_star(default_cfg, RNG_A)(outputs)
    q2 = make_q_star(default_cfg, RNG_A)(outputs)
    assert jnp.allclose(q1, q2)


def test_q_star_differs_across_seeds(default_cfg: WorldConfig) -> None:
    """Different rng seeds must produce different q* functions.

    Why: if seed changes have no effect the seed parameter is misleading and
    multi-seed analysis collapses to a single run.
    """
    outputs = all_outputs(default_cfg)
    q_a = make_q_star(default_cfg, RNG_A)(outputs)
    q_b = make_q_star(default_cfg, RNG_B)(outputs)
    assert not jnp.allclose(q_a, q_b)


def test_q_star_table_shape(default_cfg: WorldConfig) -> None:
    """q_star_table must return one score per sequence in the output space.

    Why: a shape mismatch would silently misalign scores with sequences in
    all downstream diagnostics.
    """
    table = q_star_table(default_cfg, RNG_A)
    expected_size = default_cfg.vocab_size**default_cfg.seq_len
    assert table.shape == (expected_size,)


def test_q_star_table_non_degenerate(default_cfg: WorldConfig) -> None:
    """q* must have meaningful variance across the output space.

    Why: a near-constant q* would make the experiment trivial — any policy
    would score roughly the same, and drift would be undetectable.
    """
    table = q_star_table(default_cfg, RNG_A)
    assert float(table.std()) > 0.1


def test_q_star_table_matches_pointwise(default_cfg: WorldConfig) -> None:
    """q_star_table scores must match make_q_star applied to individual sequences.

    Why: if the table and the function disagree, one of them is wrong and the
    diagnostics would report different values depending on which path is used.
    """
    outputs = all_outputs(default_cfg)
    table = q_star_table(default_cfg, RNG_A)
    q_fn = make_q_star(default_cfg, RNG_A)

    # Spot-check ten sequences at fixed indices
    for i in [0, 1, 100, 512, 1024, 2048, 4000, 4095]:
        pointwise = float(q_fn(outputs[i : i + 1])[0])
        assert abs(pointwise - float(table[i])) < 1e-5, (
            f"Mismatch at index {i}: table={float(table[i]):.6f}, "
            f"pointwise={pointwise:.6f}"
        )


# ---------------------------------------------------------------------------
# render_sequence
# ---------------------------------------------------------------------------


def test_render_sequence_default_cfg(default_cfg: WorldConfig) -> None:
    """render_sequence must return a space-separated string of compass symbols.

    Why: rendering is used in all printed output and figures; a broken renderer
    would make results unreadable without failing any numeric test.
    """
    x = jnp.array([0, 1, 2, 3])
    result = render_sequence(x)
    assert result == "↑ ↗ → ↘"


def test_render_sequence_all_symbols(default_cfg: WorldConfig) -> None:
    """Every compass symbol must be reachable via render_sequence.

    Why: a missing symbol would indicate an off-by-one in the VOCAB_SYMBOLS
    table, silently corrupting any display that relies on it.
    """
    for i, expected_symbol in enumerate(VOCAB_SYMBOLS):
        result = render_sequence(jnp.array([i]))
        assert result == expected_symbol
