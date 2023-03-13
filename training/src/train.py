#!/bin/python3
# %%
import torch
from lightning import pytorch as pl
import wandb
import warnings

from model import ReviewModel
import utils

# %%
wandb.login()
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

# %% Config
config = utils.get_config("training_config")
model_config = utils.get_model_config(config["model"], config["artifact"])
cluster_config = config["cluster"]
active_layers = config["active_layers"]
test_run = config["test_run"]
del config["cluster"], config["test_run"]

hyperparameters = {
    "weight_decay": config["optimizer"]["weight_decay"],
    "batch_size": config["batch_size"],
    "max_lr": config["max_lr"],
}
dataset_parameters = utils.get_dataset_paths(config["dataset"]["version"])
dataset_parameters["validation_split"] = config["dataset"]["validation_split"]

# %% Initialization
if not test_run:
    logger = pl.loggers.WandbLogger(
        project="rlp-t2t", entity="bsc2022-usageinfo", config=config
    )
    checkpoint_callback = utils.get_checkpoint_callback(logger)

trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_false",
    devices=cluster_config["devices"],
    num_nodes=cluster_config["num_nodes"],
    deterministic=True,
    max_epochs=config["epochs"],
    accelerator="gpu",
    callbacks=[checkpoint_callback] if not test_run else None,
    logger=logger if not test_run else None,
)

model = ReviewModel(
    model=model_config[0],
    model_name=config["model"],
    tokenizer=model_config[1],
    max_length=model_config[2],
    active_layers=active_layers,
    optimizer=utils.get_optimizer(config["optimizer"]["name"]),
    hyperparameters=hyperparameters,
    data=dataset_parameters,
    trainer=trainer,
)

# %% Training
trainer.fit(model)

# %% Testing
trainer.test(model)
