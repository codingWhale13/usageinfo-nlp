#!/usr/bin/env python3
import argparse
import os
import yaml
from typing import List
import glob
import dotenv

from review_set import ReviewSet
import helpers.label_selection as ls
from data_augmentation.configuration import get_augmentations

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
        "label_ids",
        type=str,
        help="Comma-separated list of label ids to be used for creating the dataset (e.g. 'golden_v2, gpt-3.5-turbo-leoh_v1')\nWildcards '*' and '?' are allowed",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Files to be used for creating the dataset",
    )
    parser.add_argument(
        "-a",
        "--augment_data",
        action="store_true",
        help="Apply data augmentation (interactive configuration)",
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
    label_ids,
    files,
    dataset_dir,
    **kwargs,
) -> None:
    dict_args = {
        "name": dataset_name,
        "label_ids": label_ids,
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
    label_ids = args.label_ids.split(", ")
    contains_usage_split = args.usage_split
    test_split = args.test_split
    seed = args.seed

    dataset_dir = create_dataset_dir(dataset_name=dataset_name)
    reviews = ReviewSet.from_files(*files)
    augmentation, augmentation_config = (
        get_augmentations() if args.augment_data else (None, None)
    )
    reviews, metadata = reviews.create_dataset(
        dataset_name=dataset_name,
        label_selection_strategy=ls.LabelIDSelectionStrategy(*label_ids),
        test_split=test_split,
        contains_usage_split=contains_usage_split,
        augmentation=augmentation,
        seed=seed,
    )

    reviews.save_as(os.path.join(dataset_dir, "reviews.json"))

    create_yml(
        dataset_name=dataset_name,
        label_ids=label_ids,
        files=files,
        dataset_dir=dataset_dir,
        augmentations=augmentation_config,
        **metadata,
    )


if __name__ == "__main__":
    main()
