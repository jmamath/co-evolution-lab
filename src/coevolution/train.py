"""
Co-evolution training loop and architectural variant implementations.

This file exists to orchestrate the self-rewarding loop that is the subject of
the experiment. A single outer iteration runs: (1) sample from policy,
(2) judge scores the samples, (3) policy is updated via REINFORCE with judge
score as reward, (4) judge is updated on policy-generated pairs with its own
labels as supervision.

The four architectural variants (asymmetric update rates, ensembles, periodic
reinitialization, external validation anchor) are all implemented here as
conditional branches controlled by TrainingConfig flags. This keeps the variant
logic co-located with the training loop it modifies and avoids scattered
conditional logic across multiple files.
"""

# TODO: Milestone 3
