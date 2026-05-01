"""
Shared utilities: structured logging and checkpointing.

This file exists to centralise the cross-cutting concerns that every module
needs but that do not belong to any single module. Keeping these here rather
than duplicating them across train.py and the script files ensures that log
formats and checkpoint conventions are consistent across all runs — which
matters when comparing results across seeds and variants.

Entry-point scripts (run_baseline.py, run_variants.py) call
logging.basicConfig once; all other modules obtain a module-level logger via
logging.getLogger(__name__) and never call basicConfig themselves.
"""

import json
import logging
import pathlib
import pickle

logger = logging.getLogger(__name__)


def save_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    """Append a list of records to a JSONL file.

    Creates parent directories if they do not exist. Appends to the file if it
    already exists so that runs can be resumed without losing prior history.

    Args:
        path: Destination file path. The .jsonl extension is conventional but
            not enforced.
        records: List of JSON-serialisable dicts to append, one per line.

    Side effects:
        Creates parent directories. Opens path in append mode.

    Raises:
        TypeError: If any record contains a non-serialisable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_checkpoint(path: pathlib.Path, params: dict, step: int) -> None:
    """Serialise model parameters and the current step count to disk.

    Used to persist the pretrained judge state so that all variant runs
    in M5 can start from the same initialisation rather than retraining
    independently.

    Args:
        path: Destination file path (.pkl).
        params: Flax parameter dict (nested dict of JAX arrays).
        step: Training step at which the checkpoint was taken. Stored
            alongside params so callers can verify which checkpoint they loaded.

    Side effects:
        Creates parent directories. Overwrites any existing file at path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "step": step}, f)
    logger.info("Checkpoint saved to %s at step %d", path, step)


def load_checkpoint(path: pathlib.Path) -> tuple[dict, int]:
    """Load model parameters and step count from a checkpoint file.

    Args:
        path: Path to a .pkl file created by save_checkpoint.

    Returns:
        Tuple of (params, step) where params is a Flax parameter dict and
        step is the training step recorded when the checkpoint was saved.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Checkpoint loaded from %s at step %d", path, data["step"])
    return data["params"], data["step"]
