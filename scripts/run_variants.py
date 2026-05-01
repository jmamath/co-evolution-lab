"""
Architectural variant experiments.

This script exists to test the four stabilisation mechanisms proposed in
Frontier 5 of the blog series: asymmetric update rates, judge ensembles,
periodic reinitialization, and an external validation anchor. Each variant
changes a single config flag against the baseline; all other settings are
held constant so the effect is attributable to that mechanism alone.

Run:
    python scripts/run_variants.py

Output:
    results/variants/  — one subdirectory per variant, each containing
    per-seed JSONL files and a summary CSV.

Expected wall time on M1: 2-3 hours total; runs are parallelisable across cores.
"""

import concurrent.futures
import logging
import pathlib

import jax
import jax.numpy as jnp

from coevolution.config import AgentConfig, TrainingConfig, WorldConfig
from coevolution.judge import Judge
from coevolution.policy import Policy
from coevolution.train import pretrain_judge, run_coevolution
from coevolution.world import all_outputs, q_star_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_experiment(variant_id: str, value: float, seed: int):
    """Run a single co-evolution experiment for a variant/seed combination."""
    logger.info("Starting experiment: %s=%s, seed=%d", variant_id, value, seed)

    world_cfg = WorldConfig()
    agent_cfg = AgentConfig()

    # Setup specific variant config
    cfg_kwargs = {"seed": seed, "run_name": f"{variant_id}_{value}"}
    if variant_id == "asymmetric":
        cfg_kwargs["judge_update_every"] = int(value)
    elif variant_id == "ensemble":
        cfg_kwargs["n_judges"] = int(value)
    elif variant_id == "reinit":
        cfg_kwargs["reset_every"] = int(value)
    elif variant_id == "anchor":
        cfg_kwargs["anchor_fraction"] = float(value)

    cfg = TrainingConfig(**cfg_kwargs)

    out_dir = pathlib.Path(f"results/variants/{variant_id}_{value}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-computation
    outputs_tbl = all_outputs(world_cfg)
    q_star_tbl = q_star_table(
        world_cfg, jax.random.PRNGKey(world_cfg.seed_for_q_star)
    )

    rng = jax.random.PRNGKey(seed)
    policy = Policy(cfg=agent_cfg, world=world_cfg)
    judge = Judge(cfg=agent_cfg, world=world_cfg)

    # 1. Pretrain judge(s)
    pretrained_judges_params = []
    for i in range(cfg.n_judges):
        # Use different sub-seeds for each judge in an ensemble
        rng, rng_init, rng_pretrain = jax.random.split(rng, 3)
        dummy_seq = jnp.zeros((1, world_cfg.seq_len), dtype=jnp.int32)
        params = judge.init(rng_init, dummy_seq)
        pretrained_params = pretrain_judge(
            judge, params, q_star_tbl, outputs_tbl, cfg, rng_pretrain
        )
        pretrained_judges_params.append(pretrained_params)

    # 2. Run co-evolution loop
    rng, rng_run = jax.random.split(rng)
    run_coevolution(
        policy=policy,
        judge=judge,
        world_cfg=world_cfg,
        cfg=cfg,
        pretrained_judges_params=pretrained_judges_params,
        rng=rng_run,
        results_dir=out_dir,
    )
    logger.info("Completed experiment: %s=%s, seed=%d", variant_id, value, seed)


def main():
    # Experimental grid
    variants = [
        ("asymmetric", 2),
        ("asymmetric", 5),
        ("asymmetric", 10),
        ("ensemble", 3),
        ("ensemble", 5),
        ("reinit", 3),
        ("reinit", 5),
        ("anchor", 0.05),
        ("anchor", 0.1),
    ]
    seeds = [0, 1, 2, 3, 4]

    tasks = []
    for var_id, val in variants:
        for seed in seeds:
            tasks.append((var_id, val, seed))

    logger.info("Starting variant sweep with %d runs...", len(tasks))

    # Parallelize across 4 workers to stay safe with memory/CPU on M1
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_experiment, *t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error("Experiment failed with error: %s", e)

    logger.info("Variant sweep complete.")


if __name__ == "__main__":
    main()
