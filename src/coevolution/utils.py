"""
Shared utilities: PRNG management, structured logging, and checkpointing.

This file exists to centralise the cross-cutting concerns that every module
needs but that do not belong to any single module. Keeping these here rather
than duplicating them across train.py and the script files ensures that PRNG
splitting conventions and log formats are consistent across all runs — which
matters when comparing results across seeds and variants.

Entry-point scripts (run_baseline.py, run_variants.py) call
logging.basicConfig once; all other modules obtain a module-level logger via
logging.getLogger(__name__) and never call basicConfig themselves.
"""

# TODO: implement PRNG helpers, JSONL logger, and checkpoint save/load
