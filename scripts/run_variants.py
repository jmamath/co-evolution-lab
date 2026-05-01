"""
Architectural variant experiments.

This script exists to test the five stabilisation mechanisms proposed in
Frontier 5 of the blog series: asymmetric update rates, judge ensembles,
periodic reinitialization, an external validation anchor, and a meta-judge.
Each variant changes a single config flag against the baseline; all other
settings are held constant so the effect is attributable to that mechanism alone.

Run all variants:
    python scripts/run_variants.py

Run a specific variant:
    python scripts/run_variants.py --variants meta

Run multiple variants:
    python scripts/run_variants.py --variants anchor meta

Run specific seeds only:
    python scripts/run_variants.py --variants meta --seeds 0 1 2

Output:
    results/variants/  — one subdirectory per variant setting, each containing
    per-seed JSONL files.

Expected wall time on M1: 2-3 hours for the full grid; individual variants
take ~20-30 minutes across 5 seeds.
"""

import argparse
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
    elif variant_id == "meta":
        cfg_kwargs["meta_holdout_fraction"] = float(value)

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


VARIANT_GRID = [
    ("asymmetric", 2),
    ("asymmetric", 5),
    ("asymmetric", 10),
    ("ensemble", 3),
    ("ensemble", 5),
    ("reinit", 3),
    ("reinit", 5),
    ("anchor", 0.05),
    ("anchor", 0.1),
    ("meta", 0.05),
    ("meta", 0.1),
]

ALL_SEEDS = [0, 1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser(description="Run architectural variant experiments.")
    parser.add_argument(
        "--variants",
        nargs="+",
        metavar="VARIANT",
        help="Variant IDs to run (e.g. meta anchor). Runs all if omitted.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        default=ALL_SEEDS,
        help="Seeds to run (e.g. 0 1 2). Runs all five if omitted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default: 4).",
    )
    args = parser.parse_args()

    grid = VARIANT_GRID
    if args.variants:
        unknown = set(args.variants) - {v for v, _ in grid}
        if unknown:
            parser.error(f"Unknown variant(s): {', '.join(sorted(unknown))}. "
                         f"Choose from: {', '.join(sorted({v for v, _ in grid}))}")
        grid = [(v, val) for v, val in grid if v in args.variants]

    tasks = [(var_id, val, seed) for var_id, val in grid for seed in args.seeds]
    logger.info("Starting variant sweep with %d runs...", len(tasks))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single_experiment, *t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error("Experiment failed with error: %s", e)

    logger.info("Variant sweep complete.")


if __name__ == "__main__":
    main()
