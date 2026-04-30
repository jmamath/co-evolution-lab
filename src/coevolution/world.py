"""Output space and ground-truth quality function (not visible to the judge)."""

from dataclasses import dataclass
from typing import Callable

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from coevolution.config import WorldConfig


def make_q_star(cfg: WorldConfig, rng: jax.Array) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function q*(x) -> scalar. The function is unknown to the judge."""
    rng_pos, rng_tok, rng_pair = jax.random.split(rng, 3)
    w_position = jax.random.normal(rng_pos, (cfg.seq_len,))
    w_token = jax.random.normal(rng_tok, (cfg.vocab_size,))
    # Pair interaction weights: (seq_len, vocab_size) matrix
    w_pair = jax.random.normal(rng_pair, (cfg.seq_len, cfg.vocab_size)) * cfg.pair_alpha

    def q_star(x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., seq_len) integer tokens in [0, vocab_size)
        token_scores = w_token[x]                                  # (..., seq_len)
        position_weighted = (token_scores * w_position).sum(-1)    # (...,)
        pair_scores = w_pair[jnp.arange(cfg.seq_len), x].sum(-1)   # (...,)
        return position_weighted + pair_scores

    return q_star


def all_outputs(cfg: WorldConfig) -> jnp.ndarray:
    """Return all K^L outputs as integer array of shape (K^L, L)."""
    combos = list(itertools.product(range(cfg.vocab_size), repeat=cfg.seq_len))
    return jnp.array(combos, dtype=jnp.int32)


def q_star_table(cfg: WorldConfig, rng: jax.Array) -> jnp.ndarray:
    """Precompute q*(x) for all outputs. Shape: (K^L,)."""
    q_fn = make_q_star(cfg, rng)
    outputs = all_outputs(cfg)
    return q_fn(outputs)
