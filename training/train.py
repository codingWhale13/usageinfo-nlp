#!/usr/bin/env python3
# %%
import torch
from lightning import pytorch as pl
import wandb
import warnings
import sys

from model import ReviewModel
import utils

# %%
wandb.login()
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

# Command line args
args_config = {}
for arg in sys.argv[1:]:
    if not arg.startswith("--") or "=" not in arg:
        print(f"Unrecognized argument: {arg}")
        print(
            "Please only provide arguments in the form --key=value or --key:key=value"
        )
        exit(1)

    key, value = arg[2:].split("=")
    args_config[key] = value


# %% Config
config = utils.get_config(
    args_config.pop("config", None) or utils.get_config_path("training_config")
)
for key, value in args_config.copy().items():
    try:
        if ":" in key:
            del args_config[key]
            base_key, subkey = key.split(":")
            args_config[base_key] = config[base_key] | {
                subkey: type(config[base_key][subkey])(value)
            }
        else:
            args_config[key] = type(config[key])(value)
    except (KeyError, ValueError) as e:
        if isinstance(e, KeyError):
            print(f"Unknown config key: {key}\n{e}")
            exit(1)
        else:
            print(f"Invalid value for {key}: {value}")
            print(f"Expected type: {type(config[key])}")
            exit(1)

config |= args_config

model_config = utils.get_model_config(config["model"], config["artifact"])
cluster_config = config["cluster"]
test_run = config["test_run"]
del config["cluster"], config["test_run"]

hyperparameters = {
    "weight_decay": config["optimizer"]["weight_decay"],
    "batch_size": config["batch_size"],
    "max_lr": config["max_lr"],
}
dataset_parameters = {
    "dataset_name": config["dataset"]["version"],
    "validation_split": config["dataset"]["validation_split"],
}

# %% Initialization
if not test_run:
    logger = pl.loggers.WandbLogger(
        project="rlp-t2t", entity="bsc2022-usageinfo", config=config
    )
    checkpoint_callback = utils.get_checkpoint_callback(logger, config)

trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_false",
    devices=cluster_config["devices"],
    num_nodes=cluster_config["num_nodes"],
    deterministic=True,
    max_epochs=config["epochs"],
    accelerator="auto",
    callbacks=[checkpoint_callback] if not test_run else None,
    logger=logger if not test_run else None,
)

model = ReviewModel(
    model=model_config[0],
    model_name=config["model"],
    tokenizer=model_config[1],
    max_length=model_config[2],
    active_layers=config["active_layers"],
    optimizer=utils.get_optimizer(config["optimizer"]["name"]),
    hyperparameters=hyperparameters,
    data=dataset_parameters,
    trainer=trainer,
    multiple_usage_options_strategy=config["multiple_usage_options_strategy"],
)

# %% Training
trainer.fit(model)

# %% Testing
trainer.test(model)
