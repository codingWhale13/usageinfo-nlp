#!/usr/bin/env python3
# %%
import sys
import warnings
from copy import copy
import torch
import wandb
from lightning import pytorch as pl
from pprint import pprint

from model import ReviewModel
from helpers.sustainability_logger import SustainabilityLogger
from generator import DEFAULT_GENERATION_CONFIG, Generator
from active_learning.helpers import load_active_data_module
import utils
from helpers.review_set import ReviewSet

wandb.login()

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

# Command line args
args_config = {}
for arg in sys.argv[1:]:
    if not arg.startswith("--") or "=" not in arg:
        print(f"Unrecognized argument: {arg}")
        print(
            "Please only provide arguments in the form --key=value or --key:...:key=value (for nested parameters)"
        )
        exit(1)

    key, value = arg[2:].split("=")
    args_config[key] = value


config = utils.get_config(
    args_config.pop("config", None) or utils.get_config_path("training_config")
)
for key, value in args_config.copy().items():
    try:
        if ":" in key:
            del args_config[key]
            keys = key.split(":")
            current_key = keys.pop(0)
            args_config[current_key] = copy(config[current_key])
            current_config = args_config[current_key]
            while len(keys) > 1:
                current_key = keys.pop(0)
                current_config = current_config[current_key]
            value_type = type(current_config[keys[0]])
            current_config[keys[0]] = value_type(value)
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
print("----------------------------------\nTraining config:")
pprint(config)
print("----------------------------------")

# the method below will also check if either model_name or artifact is provided
model, tokenizer, max_length, model_name = utils.initialize_model_tuple(
    config["artifact"]
    if config["artifact"]["name"] is not None
    else config["model_name"]
)

active_learning_params = config["active_learning"].get("parameters", {}) or {}

cluster_config = config["cluster"]
test_run = config["test_run"]
active_learning_module = load_active_data_module(
    config["active_learning"]["module"],
    active_learning_params,
)
files_to_generate_on = config["files_to_generate_on"]
seed = config["seed"] if config["seed"] else None
del config["cluster"], config["test_run"], config["files_to_generate_on"]

hyperparameters = {
    "weight_decay": config["optimizer"]["weight_decay"],
    "batch_size": config["batch_size"],
    "max_lr": config["optimizer"]["lr"],
}
dataset_parameters = copy(config["dataset"])
optimizer, optimizer_args = utils.get_optimizer(config["optimizer"])
pl.seed_everything(seed=config["seed"], workers=True)

if not test_run:
    logger = pl.loggers.WandbLogger(
        project="rlp-t2t", entity="bsc2022-usageinfo", config=config
    )
    checkpoint_callback = utils.get_checkpoint_callback(logger, config)

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available, using CPU instead.")

trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_false",
    devices=cluster_config["devices"],
    num_nodes=cluster_config["num_nodes"],
    deterministic=True,
    max_epochs=config["epochs"],
    accelerator="auto",
    callbacks=[checkpoint_callback] if not test_run else None,
    logger=logger if not test_run else None,
    accumulate_grad_batches=config["accumulate_grad_batches"],
)

model = ReviewModel(
    model=model,
    model_name=model_name,
    tokenizer=tokenizer,
    max_length=max_length,
    active_layers=config["active_layers"],
    optimizer=optimizer,
    optimizer_args=optimizer_args,
    hyperparameters=hyperparameters,
    dataset_config=dataset_parameters,
    trainer=trainer,
    multiple_usage_options_strategy=config["multiple_usage_options_strategy"],
    lr_scheduler_type=config["lr_scheduler_type"],
    gradual_unfreezing_mode=config["gradual_unfreezing_mode"],
    active_data_module=active_learning_module,
    prompt_id=config["prompt_id"],
)


# %% Training and testing
if not test_run:
    with SustainabilityLogger(description="training"):
        trainer.fit(model)

    with SustainabilityLogger(description="testing"):
        trainer.test()

    try:
        test_dataset = model.test_reviews

        label_id = f"model-{wandb.run.name}-auto"

        generator = Generator(
            wandb.run.name, DEFAULT_GENERATION_CONFIG, checkpoint="best"
        )
        generator.generate_label(test_dataset, label_id=label_id, verbose=True)

        test_dataset.save()
    except Exception as e:
        warnings.warn(
            "Could not generate label for the dataset. The run has probably failed.",
            e,
        )
else:
    trainer.fit(model)
    trainer.test()
