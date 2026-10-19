#!/usr/bin/env python3
"""Build the tokenized, packed training dataset once, in a single process.

TRL prepares the dataset inside `main_process_first`: rank 0 tokenizes, builds
labels and packs while the other 31 ranks wait at the first collective. For a
7.4M-row mix that is ~6.5 hours against a 30-minute process-group timeout.
Raising `ddp_timeout` does not help -- with `accelerate launch --use_deepspeed`
the group is created before TrainingArguments is read, which is why the error
still reports Timeout(ms)=1800000.

With world_size=1 there is no barrier, so the prep can take as long as it needs.
The result is saved and training loads it with skip_prepare_dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setup_logging()

    if args.out.exists() and not args.overwrite:
        logger.error("%s already exists; pass --overwrite to replace it", args.out)
        sys.exit(1)

    config = PostTrainingConfig.load(args.config)

    if getattr(config.sft, "prepared_dataset_path", None):
        logger.info("Ignoring sft.prepared_dataset_path while preparing.")
        config.sft.prepared_dataset_path = None

    run_dir = setup_run_directory(config, allow_override=False)
    logger.info("Run directory: %s", run_dir)

    logger.info("Building trainer (this is the ~6.5h part)...")
    trainer = build_trainer(config, run_dir)

    dataset = trainer.train_dataset
    logger.info("Prepared dataset: %d rows, columns %s", len(dataset), dataset.column_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.out))
    logger.info("Saved to %s", args.out)

    print(f'\nSet this in the config:\n\nsft:\n  prepared_dataset_path: "{args.out}"\n')


if __name__ == "__main__":
    main()
