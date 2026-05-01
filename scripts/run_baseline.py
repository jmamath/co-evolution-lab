"""
Baseline co-evolution experiment across multiple seeds.

This script exists to produce the headline result: ten iterations of vanilla
self-rewarding training run across five independent seeds, with diagnostics
saved at each iteration. The resulting data feeds directly into make_figures.py
to produce the 2x2 headline plot.

Run:
    python scripts/run_baseline.py --seeds 0 1 2 3 4

Output:
    results/baseline_seeds_<range>/  — one JSONL file per seed, plus a
    combined CSV used by make_figures.py.

Expected wall time on M1: 10-15 minutes for all five seeds.
"""

import argparse
import logging
import pathlib

import jax
import jax.numpy as jnp

from coevolution.config import AgentConfig, TrainingConfig, WorldConfig
from coevolution.judge import Judge
from coevolution.policy import Policy
from coevolution.train import pretrain_judge, run_coevolution
from coevolution.world import all_outputs, q_star_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not args.seeds:
        logger.error("Must provide at least one seed.")
        return

    min_seed, max_seed = min(args.seeds), max(args.seeds)
    out_dir = pathlib.Path(f"results/baseline_seeds_{min_seed}-{max_seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    world_cfg = WorldConfig()
    agent_cfg = AgentConfig()

    outputs_tbl = all_outputs(world_cfg)
    q_star_tbl = q_star_table(
        world_cfg, jax.random.PRNGKey(world_cfg.seed_for_q_star)
    )

    for seed in args.seeds:
        logger.info("=" * 60)
        logger.info("Starting baseline run for seed %d", seed)
        logger.info("=" * 60)

        rng = jax.random.PRNGKey(seed)
        cfg = TrainingConfig(seed=seed, run_name="baseline")

        policy = Policy(cfg=agent_cfg, world=world_cfg)
        judge = Judge(cfg=agent_cfg, world=world_cfg)

        # 1. Initialise judge parameters
        rng, rng_init = jax.random.split(rng)
        dummy_seq = jnp.zeros((1, world_cfg.seq_len), dtype=jnp.int32)
        judge_initial_params = judge.init(rng_init, dummy_seq)

        # 2. Pretrain judge to target alignment
        rng, rng_pretrain = jax.random.split(rng)
        pretrained_params = pretrain_judge(
            judge,
            judge_initial_params,
            q_star_tbl,
            outputs_tbl,
            cfg,
            rng_pretrain,
        )

        # 3. Run co-evolution loop
        rng, rng_run = jax.random.split(rng)
        run_coevolution(
            policy=policy,
            judge=judge,
            world_cfg=world_cfg,
            cfg=cfg,
            pretrained_judges_params=[pretrained_params],
            rng=rng_run,
            results_dir=out_dir,
        )


if __name__ == "__main__":
    main()
