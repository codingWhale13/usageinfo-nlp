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
from generator import DEFAULT_GENERATION_CONFIG, Generator
from active_learning.al_helpers import load_active_data_module
import utils
from helpers.review_set import ReviewSet
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from helpers.sustainability_tracker import SustainabilityTracker
from helpers.label_selection import (
    LabelIDSelectionStrategy,
    AbstractLabelSelectionStrategy,
    DatasetSelectionStrategy,
)
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
            args_config[current_key] = args_config.get(
                current_key, copy(config[current_key])
            )

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


if not torch.cuda.is_available():
    print("WARNING: CUDA is not available, using CPU instead.")


def initalize_training_run(
    sustainability_tracker: SustainabilityTracker,
    use_wandb_logger: bool = True,
    wandb_id: str = None,
    artifact_model_name: str = None,
    training_data_size: int = None,
) -> tuple[ReviewModel, pl.Trainer]:
    print("Initalizing new model")

    pl.seed_everything(seed=config["seed"], workers=True)

    accumalate_grad_batches = config["accumulate_grad_batches"]
    if training_data_size is not None:
        accumalate_grad_batches = min(
            accumalate_grad_batches, int(training_data_size / config["batch_size"])
        )
    log_every_n_steps = accumalate_grad_batches * 4

    if use_wandb_logger:
        print("Initializing new wandb logger")
        logger = pl.loggers.WandbLogger(
            project="rlp-t2t", entity="bsc2022-usageinfo", config=config, id=wandb_id
        )
        checkpoint_callback = utils.get_checkpoint_callback(logger, config)

    early_stopping_callback = EarlyStopping(monitor="validation_loss", patience=5)

    if artifact_model_name is not None:
        artifact = {"checkpoint": "best", "name": artifact_model_name}
    elif config["artifact"]["name"] is not None:
        artifact = config["artifact"]
    else:
        artifact = config["model_name"]

    print("Loading model artifact:", artifact)
    model, tokenizer, max_length, model_name = utils.initialize_model_tuple(artifact)

    print(
        f"Accumalting {accumalate_grad_batches} batches and logging every {log_every_n_steps} steps"
    )
    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_false",
        devices=cluster_config["devices"],
        num_nodes=cluster_config["num_nodes"],
        deterministic=True,
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=[early_stopping_callback]
        + ([checkpoint_callback] if use_wandb_logger else []),
        logger=logger if use_wandb_logger else None,
        accumulate_grad_batches=accumalate_grad_batches,
        val_check_interval=log_every_n_steps,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=None,
        max_steps=10_000,
        num_sanity_val_steps=0,
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
        sustainability_tracker=sustainability_tracker,
    )
    return model, trainer


def absolute_path_from_this_file_to(relative_path: str):
    import os

    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)


VALIDATION_REVIEW_SET = ReviewSet.from_files(
    absolute_path_from_this_file_to("../ba-30k-val-reviews.json")
)
VALIDATION_SELECTION_STRATEGY = DatasetSelectionStrategy("ba-30k-val")
TEST_REVIEW_SET = ReviewSet.from_files(
    absolute_path_from_this_file_to("../ba-30k-test-reviews.json")
)
TEST_SELECTION_STRATEGY = DatasetSelectionStrategy("ba-30k-test")

SILVER_REVIEW_SET = ReviewSet.from_files(
    absolute_path_from_this_file_to("../silver-v1.json")
)
SILVER_SELECTION_STRATEGY = LabelIDSelectionStrategy("bp-silver*")


def score_on(
    artifact_model_name,
    checkpoint,
    review_set: ReviewSet,
    reference_label_id_selection_stragey: AbstractLabelSelectionStrategy,
) -> dict:
    generator = Generator(
        artifact_model_name,
        DEFAULT_GENERATION_CONFIG,
        checkpoint=checkpoint,
        prompt_id=config["prompt_id"],
    )

    label_id = artifact_model_name
    generator.generate_label(review_set, label_id=label_id, verbose=False)
    review_set.save()
    return review_set.get_agg_scores(
        LabelIDSelectionStrategy(label_id), reference_label_id_selection_stragey
    )


active_learning_scores = []


def score_and_validate_model_on_validation(
    model: ReviewModel,
    trainer: pl.Trainer,
    artifact_model_name: str,
    checkpoint: str = None,
):
    def unpack_string_dict(str_dict: dict) -> dict:
        result = {}
        for key, value in str_dict.items():
            if type(value) == dict:
                for inner_key, inner_value in value.items():
                    result[key + "_" + inner_key] = inner_value
            else:
                result[key] = value
        return result

    print("Scoring:", artifact_model_name)

    active_learning_module.is_in_evaluation_mode = True
    validation_loss = trainer.validate(model, ckpt_path=None)[0]
    active_learning_module.is_in_evaluation_mode = False
    print("validation_loss:", validation_loss)

    active_learning_module.is_in_evaluation_mode = True
    test_loss = trainer.test(model, ckpt_path=None)[0]
    active_learning_module.is_in_evaluation_mode = False
    print("Test loss:", test_loss)

    # scores = score_on(
    #    artifact_model_name,
    #    checkpoint,
    #    VALIDATION_REVIEW_SET,
    #    VALIDATION_SELECTION_STRATEGY,
    # )

    scores = score_on(
        artifact_model_name, checkpoint, SILVER_REVIEW_SET, SILVER_SELECTION_STRATEGY
    )

    print(scores)
    if active_learning_module.iteration in active_learning_scores:
        raise ValueError(
            f"Scores for iteration {active_learning_module.iteration} already present in the scores:",
            active_learning_scores,
        )
    active_learning_scores.append(
        validation_loss
        | test_loss
        | unpack_string_dict(scores)
        | {
            "active_learning_iteration": active_learning_module.iteration,
            "acquired_training_reviews": active_learning_module.acquired_training_reviews_size(),
        }
    )
    active_learning_module.log_dataframe(
        "scores", pd.DataFrame.from_records(active_learning_scores)
    )
    return scores


# %% Training and testing
with SustainabilityTracker() as sustainability_tracker:
    sustainability_tracker.start("training")

    model, trainer = initalize_training_run(
        sustainability_tracker=sustainability_tracker, use_wandb_logger=False
    )
    if "run_name" in config and config["run_name"] is not None:
        original_run_name = config["run_name"]
    else:
        original_run_name = utils.generate_run_name()
    base_run_name = f"{original_run_name}-active_learning_dir"
    active_learning_module.base_run_name = base_run_name
    os.makedirs(
        os.path.join(
            os.getenv("MODELS", default=utils.ARTIFACT_PATH + "models"), base_run_name
        )
    )
    last_run_name = None

    # score_and_validate_model_on_validation(
    #    model, trainer, artifact_model_name=config["model_name"]
    # )
    for _ in range(config["max_active_learning_iterations"]):
        sustainability_tracker.start(
            "active_learning_iteration", active_learning_module.iteration
        )
        if active_learning_module.iteration > 0:
            current_iteration_best_model, _ = initalize_training_run(
                sustainability_tracker=sustainability_tracker,
                use_wandb_logger=False,
                artifact_model_name=last_run_name,
            )
            active_learning_module.model = current_iteration_best_model

        active_learning_module.acquire_training_reviews()

        runname = (
            f"{original_run_name}-{active_learning_module.iteration}"
            if original_run_name
            else None
        )
        model, trainer = initalize_training_run(
            sustainability_tracker,
            wandb_id=runname,
            training_data_size=active_learning_module.acquired_training_reviews_size(),
        )
        active_learning_module.model = model
        sustainability_tracker.start(
            "training_run_iteration", active_learning_module.iteration
        )
        trainer.fit(model)
        sustainability_tracker.stop(
            "training_run_iteration", active_learning_module.iteration
        )
        sustainability_tracker.stop(
            "active_learning_iteration", active_learning_module.iteration
        )
        active_learning_module.iteration += 1

        if original_run_name is None:
            original_run_name = model.run_name()
        last_run_name = model.run_name()
        wandb.finish()

        current_iteration_best_model, trainer = initalize_training_run(
            sustainability_tracker=sustainability_tracker,
            use_wandb_logger=False,
            artifact_model_name=last_run_name,
        )
        score_and_validate_model_on_validation(
            current_iteration_best_model,
            trainer,
            artifact_model_name=last_run_name,
            checkpoint="best",
        )

    sustainability_tracker.stop("training")
    sustainability_tracker.start("testing")
    trainer.test()
    sustainability_tracker.stop("testing")

exit()

try:
    label_id = f"model-{wandb.run.name}-auto"

    generator = Generator(wandb.run.name, DEFAULT_GENERATION_CONFIG, checkpoint="best")
    for file in files_to_generate_on:
        print(file)

except Exception as e:
    warnings.warn(
        "Could not generate label for the dataset. The run has probably failed.",
        e,
    )
