#!/usr/bin/env python3

import argparse
import yaml

from training.utils import get_dataset_path
from helpers.review_set import ReviewSet
import helpers.label_selection as ls
from data_augmentation.configuration import get_augmentations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively augment a dataset.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of dataset to augment",
    )
    parser.add_argument(
        "augmentation_set_name",
        type=str,
        help="Name of the augmentation set to create",
    )

    parser.format_help()
    return parser.parse_args()


def update_yml(
    dataset_name: str, augmentation_set_name: str, augmentation_config: list
) -> None:
    with open(get_dataset_path(dataset_name, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    config["augmentations"][augmentation_set_name] = augmentation_config

    with open(get_dataset_path(dataset_name, "config.yml"), "w") as f:
        config_yml = yaml.dump(config)
        print(f"Writing config.yml:\n{config_yml}")
        f.write(config_yml)


def augment_dataset(dataset_name, augmentation_set_name) -> None:
    dataset_path = get_dataset_path(dataset_name)
    reviews = ReviewSet.from_files(dataset_path)
    label_selection_strategy = ls.DatasetSelectionStrategy(dataset_name)

    for review in reviews:
        if (
            augmentation_set_name
            in review.get_label_from_strategy(label_selection_strategy)["augmentations"]
        ):
            raise Exception("Augmentation set already exists")

    augmentation, augmentation_config = get_augmentations()
    augmentation.augment(augmentation_set_name, label_selection_strategy, *reviews)
    update_yml(dataset_name, augmentation_set_name, augmentation_config)

    reviews.save()


def main():
    args = parse_args()
    print(
        f"Creating augmentation-set {args.augmentation_set_name} for dataset {args.dataset_name}..."
    )
    try:
        augment_dataset(args.dataset_name, args.augmentation_set_name)
    except Exception as e:
        print(f"Error: Could not create augmentation-set... {e}")
        exit(1)


if __name__ == "__main__":
    main()
