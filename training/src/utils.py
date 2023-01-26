import os
import glob
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration
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
}

optimizers = {
    "AdamW": torch.optim.AdamW,
}


def get_dataset_paths(dataset: str) -> dict:
    dotenv.load_dotenv()
    dataset_dir = os.path.join(
        os.getenv("DATASETS", default=ARTIFACT_PATH + "datasets"), dataset
    )

    dataset_paths = {
        "train_dataset": os.path.join(dataset_dir, "train_data.json"),
        "test_dataset": os.path.join(dataset_dir, "test_data.json"),
    }
    return dataset_paths


def get_model_path(model_artifact: dict) -> str:
    model_dirs = glob.glob(
        os.path.join(ARTIFACT_PATH, "models", f"*{model_artifact['name']}")
    )
    if len(model_dirs) == 0:
        raise ValueError("No model found with the given name")
    if len(model_dirs) > 1:
        raise ValueError("Multiple models found with the given name")

    if model_artifact["checkpoint"] is None:
        checkpoint_name = "last.ckpt"
    else:
        checkpoint_name = f"epoch={model_artifact['checkpoint']}.ckpt"

    return os.path.join(model_dirs[0], checkpoint_name)


def get_config() -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), r"../config.yml"
    )
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def get_model_config(model_name: str, model_artifact: dict)-> tuple:
    model_config = models[model_name]()
    if model_artifact["name"] is not None:
        checkpoint = torch.load(get_model_path(model_artifact))
        model_config[0].load_state_dict(
            {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        )
    return model_config

def get_optimizer(optimizer_name: str)-> torch.optim.Optimizer:
    return optimizers[optimizer_name]

def get_checkpoint_callback(logger: pl.loggers.WandbLogger):
    time = datetime.datetime.now().strftime("%m_%d_%H_%M")

    run_name = f"{time}_{logger.experiment.name}"

    return pl.callbacks.ModelCheckpoint(
        dirpath=f"/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/models/{run_name}",
        save_last=True,
        filename="{epoch}",
    )
