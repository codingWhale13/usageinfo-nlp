#!/usr/bin/env python3
import sys
import wandb

from training.train_copy import train
from training.sweep_configurations import sweep_configurations

if len(sys.argv) < 2:
    print("Usage: python sweep.py sweep_name")
    sys.exit(1)

sweep_name = sys.argv.pop(1)
sweep_configuration = sweep_configurations[sweep_name] | {"name": sweep_name}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, project="rlp-t2t", entity="bsc2022-usageinfo"
)

count = None
for arg in sys.argv[1:]:
    if arg.startswith("--count="):
        count = int(arg.split("=")[1])
        sys.argv.remove(arg)
        break

wandb.agent(sweep_id, function=lambda: train(is_sweep=True), count=count)
