#!/usr/bin/env python3
"""Build the tokenized, packed training dataset once, in a single process.

Why this exists: TRL wraps dataset preparation in `main_process_first`, so rank 0
tokenizes, builds labels and packs while the other 31 ranks wait at the first
collective. For a 7.4M-row mix that takes ~6.5 hours, and the process group's
30-minute timeout fires long before it finishes. Raising `ddp_timeout` in
TrainingArguments does not help -- with `accelerate launch --use_deepspeed` the
process group is created before TrainingArguments is read, so the observed
timeout stays at the 1800000 ms default.

Doing the prep in a job with world_size=1 removes the barrier entirely. The
result is saved with `save_to_disk`, and training loads it and tells TRL to skip
preparation. It also means the three mixing-ratio runs each pay prep once rather
than every time a job is resubmitted.

Run this with ONE process (plain `python`, not `accelerate launch`):

    python scripts/prepare_dataset.py \
        --config configs/trl/prelude-mixing-sft.yaml \
        --out /scratch/.../prepared/mixing-sft

Then set `sft.prepared_dataset_path` to that path in the config and submit as
usual.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# world_size=1 is what makes this work: no barrier, no timeout. Set before any
# torch/accelerate import so nothing picks up a stale value from the shell.
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

from post_training.config import PostTrainingConfig  # noqa: E402
from post_training.methods import build_trainer  # noqa: E402
from post_training.utils.logging import setup_logging  # noqa: E402
from post_training.utils.paths import setup_run_directory  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory to save_to_disk into")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing prepared dataset")
    args = ap.parse_args()

    setup_logging()

    if args.out.exists() and not args.overwrite:
        logger.error("%s already exists; pass --overwrite to replace it", args.out)
        sys.exit(1)

    config = PostTrainingConfig.load(args.config)

    # Prep must not read a prepared dataset, or this is circular.
    if getattr(config.sft, "prepared_dataset_path", None):
        logger.info("Ignoring sft.prepared_dataset_path while preparing.")
        config.sft.prepared_dataset_path = None

    run_dir = setup_run_directory(config, allow_override=False)
    logger.info("Run directory: %s", run_dir)

    # build_trainer runs the whole pipeline -- transforms, the assistant-mask
    # filter, tokenization, label building and packing -- so this stays correct
    # if any of those change. The model is loaded too, which is wasted work, but
    # it keeps this path identical to the training path.
    logger.info("Building trainer (this is the ~6.5h part)...")
    trainer = build_trainer(config, run_dir)

    dataset = trainer.train_dataset
    logger.info("Prepared dataset: %d rows, columns %s", len(dataset), dataset.column_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.out))
    logger.info("Saved to %s", args.out)

    print(
        f"\nNow set this in the config and submit as usual:\n"
        f'\nsft:\n  prepared_dataset_path: "{args.out}"\n'
    )


if __name__ == "__main__":
    main()
