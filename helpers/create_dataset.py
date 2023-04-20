#!/usr/bin/env python3
import argparse
import os
import yaml
from typing import List
import glob
import dotenv

from review_set import ReviewSet

DEFAULT_PATH = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/datasets"

dotenv.load_dotenv()
DATASETS_DIR = os.getenv("DATASETS", default=DEFAULT_PATH)


def arg_parse() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser(description="Create a new training dataset.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of dataset",
    )
    parser.add_argument(
        "label_id", type=str, help="Label ID to be used for creating the dataset"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Files to be used for creating the dataset",
    )
    parser.add_argument(
        "-t",
        "--test_split",
        type=float,
        default=0.1,
        help="Percentage of the data to be used for testing (default. 0.1)",
    )
    parser.add_argument(
        "-u",
        "--usage-split",
        type=float,
        default=None,
        help="Percentage of the training/validation data containing usage options (default: original)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed to shuffle data with",
    )
    return parser.parse_args(), parser.format_help()


def get_all_files(paths: list[str]) -> list[str]:
    files = []
    for path in paths:
        files.extend(glob.glob(path))
    return files


def create_dataset_dir(dataset_name: str):
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    print(f"Creating dataset at {dataset_dir}...")
    if os.path.exists(dataset_dir):
        if input("Dataset already exists. Overwrite? (y/N): ").lower() != "y":
            exit("Aborted.")
    else:
        os.mkdir(dataset_dir)

    return dataset_dir


def create_yml(
    dataset_name,
    label_id,
    files,
    dataset_dir,
    **kwargs,
) -> None:
    dict_args = {
        "name": dataset_name,
        "label_id": label_id,
        "files": files,
    } | kwargs

    with open(os.path.join(dataset_dir, "config.yml"), "w") as file:
        config = yaml.dump(dict_args)
        print(config, end="")
        file.write(config)


def main():
    args, _ = arg_parse()
    files = get_all_files(args.files)
    dataset_name = args.dataset_name
    label_id = args.label_id
    contains_usage_split = args.usage_split
    test_split = args.test_split
    seed = args.seed

    dataset_dir = create_dataset_dir(dataset_name=dataset_name)
    reviews = ReviewSet.from_files(*files)
    actual_ratios = reviews.create_dataset(
        dataset_name=dataset_name,
        label_id=label_id,
        test_split=test_split,
        contains_usage_split=contains_usage_split,
        seed=seed,
    )

    reviews.save_as(os.path.join(dataset_dir, "reviews.json"))

    create_yml(
        dataset_name=dataset_name,
        label_id=label_id,
        files=files,
        dataset_dir=dataset_dir,
        **actual_ratios,
    )


if __name__ == "__main__":
    main()
