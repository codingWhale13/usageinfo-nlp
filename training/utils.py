import os
import glob
import yaml
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    optimization,
)
import torch
import dotenv
import datetime
from lightning import pytorch as pl

ARTIFACT_PATH = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/"

models = {
    "t5-small": lambda: (
        T5ForConditionalGeneration.from_pretrained("t5-small"),
        T5Tokenizer.from_pretrained("t5-small", model_max_length=512),
        512,
    ),
    "t5-base": lambda: (
        T5ForConditionalGeneration.from_pretrained("t5-base"),
        T5Tokenizer.from_pretrained("t5-base", model_max_length=512),
        512,
    ),
    "t5-large": lambda: (
        T5ForConditionalGeneration.from_pretrained("t5-large"),
        T5Tokenizer.from_pretrained("t5-large", model_max_length=512),
        512,
    ),
    "bart-base": lambda: (
        BartForConditionalGeneration.from_pretrained("facebook/bart-base"),
        BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=1024),
        1024,
    ),
    "flan-t5-base": lambda: (
        T5ForConditionalGeneration.from_pretrained("google/flan-t5-base"),
        T5Tokenizer.from_pretrained("google/flan-t5-base", model_max_length=512),
        512,
    ),
}

optimizers = {
    "AdamW": (torch.optim.AdamW, ["weight_decay"]),
    "AdaFactor": (
        optimization.Adafactor,
        ["scale_parameter", "relative_step", "warmup_init", "lr"],
    ),
}


def get_dataset_path(dataset: str, review_set_name: str = "reviews.json") -> str:
    dotenv.load_dotenv()
    dataset_dir = os.path.join(
        os.getenv("DATASETS", default=ARTIFACT_PATH + "datasets"), dataset
    )

    return os.path.join(dataset_dir, review_set_name)


def get_model_path(model_artifact: dict) -> str:
    if model_artifact["checkpoint"] is None:
        checkpoint_name = "last.ckpt"
    else:
        checkpoint_name = f"epoch={model_artifact['checkpoint']}.ckpt"

    return os.path.join(get_model_dir(model_artifact["name"]), checkpoint_name)


def get_model_review_set_path(artifact_name: str):
    return os.path.join(get_model_dir(artifact_name), "reviews.json")


def get_config_path(name: str) -> dict:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name + ".yml")


def get_config(path: str) -> dict:
    print(f"Loading config from {os.path.abspath(path)}")
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_model_dir(artifact_name: str) -> str:
    model_dirs = glob.glob(os.path.join(ARTIFACT_PATH, "models", f"*{artifact_name}"))
    if len(model_dirs) == 0:
        raise ValueError("No model found with the given name")
    if len(model_dirs) > 1:
        raise ValueError("Multiple models found with the given name")
    # We know there is only one model dir, so we take the first one
    return model_dirs[0]


def get_config_from_artifact(artifact_name: str) -> dict:
    model_dir = get_model_dir(artifact_name)
    with open(os.path.join(model_dir, "config.yml"), "r") as file:
        return yaml.safe_load(file)


def get_model_config(model_name: str, model_artifact: dict) -> tuple:
    checkpoint = None
    if model_artifact["name"] is not None:
        checkpoint = torch.load(get_model_path(model_artifact))
    return get_model_config_from_checkpoint(model_name, checkpoint)


def get_model_config_from_checkpoint(model_name: str, checkpoint: dict) -> tuple:
    model_config = models[model_name]()
    if checkpoint is not None:
        model_config[0].load_state_dict(
            {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        )
    return model_config


def get_optimizer(optimizer_args: dict) -> torch.optim.Optimizer:
    optimizer, allowed_args = optimizers[optimizer_args["name"]]
    optimizer_args = {
        k: v for k, v in optimizer_args.items() if k in allowed_args and v != None
    }
    return optimizer, optimizer_args


def get_checkpoint_callback(logger: pl.loggers.WandbLogger, config):
    time = datetime.datetime.now().strftime("%m_%d_%H_%M")

    run_name = f"{time}_{logger.experiment.name}"
    dirpath = f"/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/models/{run_name}"

    os.mkdir(dirpath)
    with open(os.path.join(dirpath, "config.yml"), "w+") as file:
        yaml.dump(config, file)

    return pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        save_last=True,
        filename="{epoch}",
        save_weights_only=True,
        save_top_k=2,
        monitor="epoch_val_loss",
    )
