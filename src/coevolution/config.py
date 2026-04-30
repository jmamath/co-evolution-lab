from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldConfig:
    vocab_size: int = 8
    seq_len: int = 4
    pair_alpha: float = 0.3
    seed_for_q_star: int = 42


@dataclass(frozen=True)
class AgentConfig:
    embed_dim: int = 16
    hidden_dim: int = 32
    judge_init_alignment: float = 0.7


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    n_iterations: int = 10
    steps_per_iter: int = 200
    batch_size: int = 64
    lr_policy: float = 3e-4
    lr_judge: float = 3e-4
    entropy_bonus: float = 0.01
    judge_init_alignment: float = 0.7
    # Variant knobs
    judge_update_every: int = 1
    n_judges: int = 1
    reset_every: int = 0          # 0 = never
    anchor_fraction: float = 0.0
    run_name: str = "baseline"
