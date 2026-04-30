"""Categorical autoregressive policy: embedding -> GRU -> logits per step."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from coevolution.config import AgentConfig, WorldConfig


class Policy(nn.Module):
    cfg: AgentConfig
    world: WorldConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for training.

        Args:
            x: integer tokens of shape (batch, seq_len). Token at position t is
               the *input* to produce logits for position t+1; position 0 uses
               a learned start embedding.

        Returns:
            logits of shape (batch, seq_len, vocab_size).
        """
        B, L = x.shape
        emb = nn.Embed(self.world.vocab_size + 1, self.cfg.embed_dim, name="embed")

        # Start token index = vocab_size (one beyond normal tokens)
        start = jnp.full((B, 1), self.world.vocab_size, dtype=jnp.int32)
        inp = jnp.concatenate([start, x[:, :-1]], axis=1)  # (B, L)
        embedded = emb(inp)  # (B, L, embed_dim)

        gru = nn.GRUCell(self.cfg.hidden_dim, name="gru")
        h = jnp.zeros((B, self.cfg.hidden_dim))
        logit_proj = nn.Dense(self.world.vocab_size, name="proj")

        all_logits = []
        for t in range(L):
            h, _ = gru(h, embedded[:, t])
            all_logits.append(logit_proj(h))

        return jnp.stack(all_logits, axis=1)  # (B, L, vocab_size)

    def log_prob(self, params, x: jnp.ndarray) -> jnp.ndarray:
        """Log probability of sequence x under the policy. Shape: (batch,)."""
        logits = self.apply(params, x)  # (B, L, V)
        lp = jax.nn.log_softmax(logits, axis=-1)  # (B, L, V)
        token_lp = lp[jnp.arange(x.shape[0])[:, None], jnp.arange(x.shape[1])[None, :], x]
        return token_lp.sum(-1)

    def sample(self, params, rng: jax.Array, n: int) -> jnp.ndarray:
        """Sample n sequences autoregressively. Shape: (n, seq_len)."""
        L = self.world.seq_len
        gru = nn.GRUCell(self.cfg.hidden_dim, name="gru")
        emb = nn.Embed(self.world.vocab_size + 1, self.cfg.embed_dim, name="embed")

        def body(carry, t):
            h, prev_tok, rng = carry
            e = self.apply(params, prev_tok, method=lambda m, tok: m.embed_token(tok))
            new_h, _ = self.apply(params, h, e, method=lambda m, h, e: m.gru_step(h, e))
            logits = self.apply(params, new_h, method=lambda m, h: m.project(h))
            rng, sub = jax.random.split(rng)
            tok = jax.random.categorical(sub, logits)
            return (new_h, tok, rng), tok

        # Use a simpler manual loop via apply on the whole model
        return self._sample_loop(params, rng, n)

    def _sample_loop(self, params, rng: jax.Array, n: int) -> jnp.ndarray:
        """Ancestral sampling loop, returns (n, seq_len) tokens."""
        L = self.world.seq_len
        V = self.world.vocab_size
        h = jnp.zeros((n, self.cfg.hidden_dim))

        # Extract sub-modules via bound apply calls
        def embed(tok):
            return self.apply(params, tok, method=lambda m, t: m.embed_single(t))

        def step(h, e):
            return self.apply(params, h, e, method=lambda m, h, e: m.gru_step_single(h, e))

        def proj(h):
            return self.apply(params, h, method=lambda m, h: m.proj_single(h))

        start = jnp.full((n,), V, dtype=jnp.int32)
        e = embed(start)
        tokens = []
        for _ in range(L):
            h, _ = step(h, e)
            logits = proj(h)
            rng, sub = jax.random.split(rng)
            tok = jax.random.categorical(sub, logits)
            tokens.append(tok)
            e = embed(tok)

        return jnp.stack(tokens, axis=1)  # (n, L)

    def embed_single(self, tok: jnp.ndarray) -> jnp.ndarray:
        return nn.Embed(self.world.vocab_size + 1, self.cfg.embed_dim, name="embed")(tok)

    def gru_step_single(self, h, e):
        return nn.GRUCell(self.cfg.hidden_dim, name="gru")(h, e)

    def proj_single(self, h):
        return nn.Dense(self.world.vocab_size, name="proj")(h)
