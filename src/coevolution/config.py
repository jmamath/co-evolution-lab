"""
Configuration dataclasses for the co-evolution experiment.

This file exists as the single source of truth for all tuneable parameters.
Keeping configs frozen dataclasses (rather than dicts or argparse namespaces)
gives us immutability, IDE autocomplete, and a stable serialisation target —
important when logging results across many seeds and variants.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    """Defines the synthetic output space and ground-truth quality function.

    Attributes:
        vocab_size: Number of distinct tokens (K). Determines the width of the
            output alphabet. Default 8 maps to the 8 compass symbols.
        seq_len: Length of each output sequence (L). Combined with vocab_size,
            the full output space has vocab_size ** seq_len entries.
        pair_alpha: Scale of the pair-interaction term in q*. Kept small so the
            interaction is a secondary effect rather than dominating the signal.
        seed_for_q_star: RNG seed used to draw the fixed q* weights. Pinned so
            q* is identical across all experimental runs and seeds.
    """

    vocab_size: int = 8
    seq_len: int = 4
    pair_alpha: float = 0.3
    seed_for_q_star: int = 42


@dataclass(frozen=True)
class AgentConfig:
    """Shared architectural hyperparameters for the policy and judge networks.

    Both networks use the same embedding and hidden dimensions so the parameter
    counts are comparable and the experiment is not biased toward one side.

    Attributes:
        embed_dim: Dimensionality of the token embedding.
        hidden_dim: Dimensionality of the GRU hidden state.
        judge_init_alignment: Target Spearman correlation with q* at which
            judge pre-training stops. 0.7 puts the judge in the interesting
            regime: good enough to give a meaningful learning signal, imperfect
            enough to drift under co-evolution pressure.
    """

    embed_dim: int = 16
    hidden_dim: int = 32
    judge_init_alignment: float = 0.7


@dataclass(frozen=True)
class TrainingConfig:
    """Controls the co-evolution training loop and all architectural variants.

    Baseline behaviour is recovered when all variant knobs are at their default
    values. Each variant changes exactly one knob; the rest stay at baseline.

    Attributes:
        seed: RNG seed for this run. Seeds 0-4 are used for the headline plot.
        n_iterations: Number of outer co-evolution iterations.
        steps_per_iter: Number of inner gradient steps per iteration.
        batch_size: Number of sequences sampled from the policy per inner step.
        lr_policy: Learning rate for the policy optimiser.
        lr_judge: Learning rate for the judge optimiser.
        entropy_bonus: Coefficient on the policy entropy term in the REINFORCE
            objective. Prevents premature distributional collapse.
        judge_init_alignment: Forwarded to judge pre-training; see AgentConfig.
        judge_update_every: Variant A — update the judge only every N policy
            steps. Values > 1 decouple the two update rates.
        n_judges: Variant B — ensemble size. Policy reward is the mean score
            across all judges.
        reset_every: Variant C — reset judge to its pre-trained state every N
            outer iterations. 0 means never reset.
        anchor_fraction: Variant D — fraction of each judge-update batch drawn
            uniformly from the full output space and labelled with q* directly.
        run_name: Label used for result directories and log prefixes.
    """

    seed: int = 0
    n_iterations: int = 10
    steps_per_iter: int = 200
    batch_size: int = 64
    lr_policy: float = 3e-4
    lr_judge: float = 3e-4
    entropy_bonus: float = 0.01
    judge_init_alignment: float = 0.7
    judge_update_every: int = 1
    n_judges: int = 1
    reset_every: int = 0
    anchor_fraction: float = 0.0
    run_name: str = "baseline"
