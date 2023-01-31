#!/bin/python3
# %%
import torch
from lightning import pytorch as pl
import wandb
import warnings

import model, utils

# %%
wandb.login()
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

# %% Config
config = utils.get_config()
model_config = utils.get_model_config(config["model"], config["artifact"])
cluster_config = config["cluster"]
del config["cluster"]

hyperparameters = {
    key: config["optimizer"][key] for key in ["learning_rate", "weight_decay"]
}
hyperparameters["batch_size"] = config["batch_size"]
dataset_parameters = utils.get_dataset_paths(config["dataset"]["version"])
dataset_parameters["validation_split"] = config["dataset"]["validation_split"]

# %% Initialization
model = model.ReviewModel(
    model=model_config[0],
    model_name=config["model"],
    tokenizer=model_config[1],
    max_length=model_config[2],
    optimizer=utils.get_optimizer(config["optimizer"]["name"]),
    hparameters=hyperparameters,
    data=dataset_parameters,
)

logger = pl.loggers.WandbLogger(
    project="rlp-t2t", entity="bsc2022-usageinfo", config=config
)

checkpoint_callback = utils.get_checkpoint_callback(logger)

trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_false",
    devices=cluster_config["devices"],
    num_nodes=cluster_config["num_nodes"],
    deterministic=True,
    logger=logger,
    max_epochs=config["epochs"],
    accelerator="gpu",
    callbacks=[checkpoint_callback],
)

# %% Training
trainer.fit(model)

# %% Testing
trainer.test(model)
