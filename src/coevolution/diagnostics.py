"""
Exact diagnostic metrics computed by enumeration over the full output space.

This file exists to measure what is actually happening during co-evolution, as
opposed to what the training loop thinks is happening. All four metrics are
computed exactly — no sampling noise — because the output space (4096 sequences)
is fully enumerable. This is the key advantage of toy scale: the gap between
judge-perceived progress and ground-truth progress can be measured precisely
rather than estimated.

The four headline metrics are:
  - policy_quality:          E_x~π[q*(x)]          — true quality under policy
  - judge_alignment:         Spearman(J(x), q*(x))  — judge's fidelity to ground truth
  - policy_entropy:          H(π)                   — distributional spread
  - judge_perceived_quality: E_x~π[J(x)]            — what the loop optimises

The contrast between policy_quality and judge_perceived_quality is the visual
punchline of the project: the latter climbs while the former plateaus.
"""

# TODO: Milestone 4
