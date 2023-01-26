#!/bin/python3
# %%
import torch
from lightning import pytorch as pl
import wandb

import model, generator, utils

# %%
wandb.login()
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)

# %% Config
config = utils.get_config()
model_config = utils.get_model_config(config["model"], config["artifact"])

hyperparameters = {
    key: config["optimizer"][key] for key in ["learning_rate", "weight_decay"]
}
hyperparameters["batch_size"] = config["batch_size"]
dataset_parameters = utils.get_dataset_paths(config["dataset"]["version"])
dataset_parameters["validation_split"] = config["dataset"]["validation_split"]
model_generator = generator.Generator(tokenizer= model_config[1], max_length=model_config[2])

# %% Initialization
model = model.ReviewModel(
    model=model_config[0],
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
    deterministic=True,
    logger=logger,
    max_epochs=config["epochs"],
    devices=1,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
)

# %% Training
trainer.fit(model)

# %% Testing
trainer.test(model)

# %% Model Generation
model_generator.generate(model=model)