"""
Categorical autoregressive policy network.

This file exists to define how the policy generates token sequences. The policy
is the agent under training: it starts with a roughly uniform distribution over
the 4096 possible sequences and is gradually shaped by the judge's reward signal.
Tracking how its distribution changes — and whether those changes reflect genuine
quality improvement — is the central question of the experiment.

Architecture: token embedding -> 1-layer GRU -> linear projection to vocab logits,
one set of logits per position. Around 5K parameters total.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from coevolution.config import AgentConfig, WorldConfig


class Policy(nn.Module):
    """Autoregressive policy over fixed-length token sequences.

    Generates sequences of length world.seq_len from a vocabulary of size
    world.vocab_size. A learned start-of-sequence token (index vocab_size)
    seeds the GRU at position 0.

    Attributes:
        cfg: Architecture hyperparameters (embed_dim, hidden_dim).
        world: World configuration (vocab_size, seq_len).
    """

    cfg: AgentConfig
    world: WorldConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute per-position logits for a batch of sequences.

        Teacher-forces the input: at each position t the GRU receives the
        embedding of x[t-1] (with a learned SOS token at t=0) and produces
        logits for position t.

        Args:
            x: Integer token array of shape (batch, seq_len) with values in
               [0, vocab_size).

        Returns:
            Float array of shape (batch, seq_len, vocab_size) containing
            unnormalised log-probabilities for each position.
        """
        B, L = x.shape
        emb = nn.Embed(self.world.vocab_size + 1, self.cfg.embed_dim, name="embed")

        # SOS index sits one beyond the normal vocabulary so it never collides
        # with real tokens and has its own learned embedding.
        sos = jnp.full((B, 1), self.world.vocab_size, dtype=jnp.int32)
        inp = jnp.concatenate([sos, x[:, :-1]], axis=1)  # (B, L)
        embedded = emb(inp)  # (B, L, embed_dim)

        gru = nn.GRUCell(self.cfg.hidden_dim, name="gru")
        proj = nn.Dense(self.world.vocab_size, name="proj")
        h = jnp.zeros((B, self.cfg.hidden_dim))

        logits_per_step = []
        for t in range(L):
            h, _ = gru(h, embedded[:, t])
            logits_per_step.append(proj(h))

        return jnp.stack(logits_per_step, axis=1)  # (B, L, vocab_size)

    def log_prob(self, params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log probability of each sequence under the current policy.

        Args:
            params: Flax parameter dict returned by init.
            x: Integer token array of shape (batch, seq_len).

        Returns:
            Float array of shape (batch,) containing sum log-probabilities.
        """
        logits = self.apply(params, x)  # (B, L, V)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, L, V)
        batch_idx = jnp.arange(x.shape[0])[:, None]
        pos_idx = jnp.arange(x.shape[1])[None, :]
        token_log_probs = log_probs[batch_idx, pos_idx, x]  # (B, L)
        return token_log_probs.sum(-1)  # (B,)

    def sample(self, params: dict, rng: jax.Array, n: int) -> jnp.ndarray:
        """Sample n sequences autoregressively from the current policy.

        At each step the token sampled at step t-1 is fed back as input for
        step t (unlike __call__, which teacher-forces the gold sequence).

        Args:
            params: Flax parameter dict returned by init.
            rng: JAX PRNG key; split internally across decoding steps.
            n: Number of sequences to sample in parallel.

        Returns:
            Integer array of shape (n, seq_len) with values in [0, vocab_size).
        """
        L = self.world.seq_len
        h = jnp.zeros((n, self.cfg.hidden_dim))

        sos = jnp.full((n,), self.world.vocab_size, dtype=jnp.int32)
        current_token = sos
        sampled_tokens = []

        for _ in range(L):
            e = self.apply(params, current_token, method=lambda m, t: m._embed(t))
            h, _ = self.apply(params, h, e, method=lambda m, h, e: m._gru_step(h, e))
            logits = self.apply(params, h, method=lambda m, h: m._project(h))
            rng, step_rng = jax.random.split(rng)
            current_token = jax.random.categorical(step_rng, logits)
            sampled_tokens.append(current_token)

        return jnp.stack(sampled_tokens, axis=1)  # (n, L)

    # ------------------------------------------------------------------
    # Sub-module helpers used by sample() to call individual layers
    # through Flax's apply interface without re-running the full forward
    # pass. Each helper reuses the same named layers as __call__ so the
    # shared parameter dict is consistent.
    # ------------------------------------------------------------------

    def _embed(self, tok: jnp.ndarray) -> jnp.ndarray:
        """Look up token embeddings.

        Args:
            tok: Integer array of shape (batch,).

        Returns:
            Float array of shape (batch, embed_dim).
        """
        return nn.Embed(self.world.vocab_size + 1, self.cfg.embed_dim, name="embed")(tok)

    def _gru_step(self, h: jnp.ndarray, e: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Advance the GRU by one step.

        Args:
            h: Hidden state of shape (batch, hidden_dim).
            e: Input embedding of shape (batch, embed_dim).

        Returns:
            Tuple of (new_hidden_state, new_hidden_state) matching Flax GRUCell
            convention where both elements of the tuple are the new state.
        """
        return nn.GRUCell(self.cfg.hidden_dim, name="gru")(h, e)

    def _project(self, h: jnp.ndarray) -> jnp.ndarray:
        """Project hidden state to vocabulary logits.

        Args:
            h: Hidden state of shape (batch, hidden_dim).

        Returns:
            Float array of shape (batch, vocab_size).
        """
        return nn.Dense(self.world.vocab_size, name="proj")(h)
