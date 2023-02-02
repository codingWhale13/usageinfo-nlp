import argparse
import os

from generator import Generator

DEFAULT_PATH = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/datasets"
DATASETS_DIR = os.getenv("DATASETS", default=DEFAULT_PATH)


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Predict usage options on given dataset using a trained model."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of dataset version",
    )
    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name of model artifact to use (wandb run name)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint to use for prediction (default is last)",
    )
    return parser.parse_args(), parser.format_help()


def main():
    args, _ = arg_parse()
    dataset_version = args.dataset_name
    artifact_name = args.artifact_name
    checkpoint = args.checkpoint

    generator = Generator(artifact_name, dataset_version, checkpoint)
    generator.generate()


if __name__ == "__main__":
    main()