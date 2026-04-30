"""
Synthetic output space and ground-truth quality function q*.

This file exists to define the environment that the policy and judge operate in.
q* is the true measure of sequence quality; it is visible only to the diagnostics
layer, never to the judge. This separation is what makes the alignment-drift
phenomenon meaningful: the judge must approximate q* from experience, and we can
measure exactly how that approximation degrades over co-evolution iterations.

The full output space (vocab_size ** seq_len = 4096 sequences by default) is
small enough to enumerate exactly, which is why all diagnostics can be computed
without Monte Carlo sampling.
"""

import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from coevolution.config import WorldConfig

# Human-readable labels for the 8 vocabulary tokens (compass directions).
# Index i maps to VOCAB_SYMBOLS[i] for display; internal computations use ints.
VOCAB_SYMBOLS: list[str] = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]


def make_q_star(cfg: WorldConfig, rng: jax.Array) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build and return the ground-truth quality function for a given world config.

    The returned function is a closure over fixed random weights drawn from rng.
    It is deterministic for a given (cfg, rng) pair and must never be passed to
    the judge — it is the oracle used only by diagnostics.

    q*(x) = sum_i( w_position[i] * w_token[x[i]] )  +  sum_i( w_pair[i, x[i]] )

    The pair term (scaled by cfg.pair_alpha) ensures q* is compositional: certain
    (position, token) combinations are worth more than the sum of their individual
    weights. This prevents the judge from recovering q* through independent
    per-position or per-token statistics alone.

    Args:
        cfg: World configuration specifying vocab_size, seq_len, and pair_alpha.
        rng: JAX PRNG key used to draw the fixed weight vectors. Split internally;
            the caller's key is not mutated.

    Returns:
        A function q_star(x) that maps integer token arrays of shape (..., seq_len)
        to scalar quality scores of shape (...,).

    Raises:
        ValueError: If cfg.vocab_size != len(PLANET_SYMBOLS) when the default
            symbol table is in use (caught at call time by callers that render
            sequences).
    """
    rng_pos, rng_tok, rng_pair = jax.random.split(rng, 3)
    w_position: jnp.ndarray = jax.random.normal(rng_pos, (cfg.seq_len,))
    w_token: jnp.ndarray = jax.random.normal(rng_tok, (cfg.vocab_size,))
    # pair_alpha keeps interaction scores secondary to positional token scores
    w_pair: jnp.ndarray = jax.random.normal(rng_pair, (cfg.seq_len, cfg.vocab_size)) * cfg.pair_alpha

    def q_star(x: jnp.ndarray) -> jnp.ndarray:
        token_scores = w_token[x]
        position_weighted = (token_scores * w_position).sum(-1)
        pair_scores = w_pair[jnp.arange(cfg.seq_len), x].sum(-1)
        return position_weighted + pair_scores

    return q_star


def all_outputs(cfg: WorldConfig) -> jnp.ndarray:
    """Enumerate every possible output sequence in the world.

    Args:
        cfg: World configuration specifying vocab_size and seq_len.

    Returns:
        Integer array of shape (vocab_size ** seq_len, seq_len) containing
        every token sequence in lexicographic order.
    """
    combos = list(itertools.product(range(cfg.vocab_size), repeat=cfg.seq_len))
    return jnp.array(combos, dtype=jnp.int32)


def q_star_table(cfg: WorldConfig, rng: jax.Array) -> jnp.ndarray:
    """Precompute q*(x) for every sequence in the output space.

    This table is the diagnostic oracle. It is computed once at the start of a
    run and passed into diagnostics; it is never exposed to the judge.

    Args:
        cfg: World configuration.
        rng: JAX PRNG key forwarded to make_q_star.

    Returns:
        Float array of shape (vocab_size ** seq_len,) where entry i holds the
        ground-truth quality of the i-th sequence returned by all_outputs(cfg).
    """
    q_fn = make_q_star(cfg, rng)
    outputs = all_outputs(cfg)
    return q_fn(outputs)


def render_sequence(x: jnp.ndarray) -> str:
    """Convert an integer token sequence to its planet-symbol representation.

    Args:
        x: 1-D integer array of length seq_len with values in [0, vocab_size).

    Returns:
        A string of compass symbols, e.g. '↑ ↗ ↓ ←'.

    Raises:
        IndexError: If any token index exceeds len(VOCAB_SYMBOLS) - 1.
    """
    return " ".join(VOCAB_SYMBOLS[int(t)] for t in x)
