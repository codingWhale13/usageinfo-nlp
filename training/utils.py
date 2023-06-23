import os
import glob
from functools import singledispatch
import yaml
from transformers import (
    RwkvForCausalLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    optimization,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
)
import torch
import dotenv
import datetime
from lightning import pytorch as pl
from typing import Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ARTIFACT_PATH = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/"

model_tuples = {
    "t5-small": lambda: (
        T5ForConditionalGeneration.from_pretrained("t5-small"),
        T5Tokenizer.from_pretrained("t5-small", model_max_length=512),
        512,
        True,
    ),
    "t5-base": lambda: (
        T5ForConditionalGeneration.from_pretrained("t5-base"),
        T5Tokenizer.from_pretrained("t5-base", model_max_length=512),
        512,
        True,
    ),
    "t5-large": lambda: (
        T5ForConditionalGeneration.from_pretrained("t5-large"),
        T5Tokenizer.from_pretrained("t5-large", model_max_length=512),
        512,
        True,
    ),
    "bart-base": lambda: (
        BartForConditionalGeneration.from_pretrained("facebook/bart-base"),
        BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=1024),
        1024,
        True,
    ),
    "flan-t5-base": lambda: (
        T5ForConditionalGeneration.from_pretrained("google/flan-t5-base"),
        T5Tokenizer.from_pretrained("google/flan-t5-base", model_max_length=512),
        512,
        True,
    ),
    "RWKV": lambda: (
        RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile"),
        AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile", model_max_length=512),
        512,
        False,
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


def get_model_artifact_path(model_artifact: dict) -> str:
    checkpoint_name = model_artifact["checkpoint"]
    if checkpoint_name is None:
        checkpoint_file_name = "last.ckpt"
    elif isinstance(checkpoint_name, int) or checkpoint_name.isdigit():
        checkpoint_file_name = f"epoch={checkpoint_name}.ckpt"
    else:
        checkpoint_file_name = f"{checkpoint_name}.ckpt"

    checkpoint_path = os.path.join(
        get_model_dir(model_artifact["name"]), checkpoint_file_name
    )
    if not os.path.exists(checkpoint_path):
        raise ValueError(
            f"Checkpoint {checkpoint_name} does not exist for artifact {model_artifact['name']}"
        )
    return checkpoint_path


def get_config_path(name: str) -> dict:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name + ".yml")


def get_config(path: str) -> dict:
    print(f"Loading config from {os.path.abspath(path)}")
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_model_dir_file_path(artifact_name: str, file_name: str):
    return os.path.join(get_model_dir(artifact_name), file_name)


def get_model_dir(artifact_name: str) -> str:
    model_dirs = glob.glob(os.path.join(ARTIFACT_PATH, "models", f"*{artifact_name}"))
    if len(model_dirs) == 0:
        raise ValueError("No model found with the given name")
    if len(model_dirs) > 1:
        raise ValueError("Multiple models found with the given name")
    # We know there is only one model dir, so we take the first one
    return model_dirs[0]


def load_config_from_artifact_name(artifact_name: str) -> dict:
    model_dir = get_model_dir(artifact_name)
    with open(os.path.join(model_dir, "config.yml"), "r") as file:
        return yaml.safe_load(file)


@singledispatch
def initialize_model_tuple(model):
    """Returns a tuple of model, tokenizer, max_length, model_name."""
    raise NotImplementedError(
        f"model must be string or artifact dict, not {type(model)}"
    )


@initialize_model_tuple.register(str)
def _(model_name: str):
    """Returns a tuple of model, tokenizer, max_length, model_name.

    Args:
        model_name (str): name of model type (i.e. t5-small)
    """
    return model_tuples[model_name]() + (model_name,)


@initialize_model_tuple.register(dict)
def _(artifact: dict):
    """Returns a tuple of model, tokenizer, max_length, model_name.

    Args:
        artifact (dict): dictionary containing the name (wandb run name)
            and checkpoint (i.e. "best")
    """
    checkpoint = torch.load(get_model_artifact_path(artifact))
    model_name = checkpoint.get("model_name", checkpoint.get("model"))
    model_tuple = model_tuples[model_name]()
    model_tuple[0].load_state_dict(
        {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    )
    return model_tuple + (model_name,)


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
        filename="best",
        save_weights_only=True,
        monitor="epoch_val_loss",
    )


def freeze_model(active_layers: dict, model) -> Tuple[int, int]:
    def unfreeze(component, slice_: str) -> int:
        transformer_blocks = eval(f"list(component.block)[{slice_}]")
        for block in transformer_blocks:
            for param in block.parameters():
                param.requires_grad = True
        # Returns the number of unfrozen transformer blocks
        return len(transformer_blocks)

    for param in model.parameters():
        param.requires_grad = False

    if active_layers["lm_head"]:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    active_encoder_layers = unfreeze(model.encoder, active_layers["encoder"])
    active_decoder_layers = unfreeze(model.decoder, active_layers["decoder"])
    return active_encoder_layers, active_decoder_layers


def gradual_unfreeze(
    model,
    epoch: int,
    gradual_unfreezing_mode,
    active_encoder_layers,
    active_decoder_layers,
):
    def unfreeze(blocks, epoch: int):
        for i in range(1, len(blocks) + 1):
            if epoch >= i:
                for param in blocks[len(blocks) - i].parameters():
                    param.requires_grad = True

    def unfreeze_helper(active_layers, epoch, speed, module):
        epoch = epoch // speed + active_layers
        unfreeze(module, epoch)

    if gradual_unfreezing_mode is not None:
        unfreezing_modes = gradual_unfreezing_mode.split(", ")
        for mode in unfreezing_modes:
            if "encoder" in mode:
                unfreeze_helper(
                    active_encoder_layers,
                    epoch,
                    int(mode.split(" ")[1]),
                    model.encoder.block,
                )

            if "decoder" in mode:
                unfreeze_helper(
                    active_decoder_layers,
                    epoch,
                    int(mode.split(" ")[1]),
                    model.decoder.block,
                )
