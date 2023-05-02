#!/usr/bin/env python3
import argparse

from training.generator import Generator
from training import utils
from helpers.review_set import ReviewSet
from helpers.label_selection import DatasetSelectionStrategy


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Predict usage options on given reviewset files with a trained artifact."
    )
    parser.add_argument(
        "-g",
        "--generation_config",
        type=str,
        default=utils.get_config_path("generation_config"),
        help="Path to generation config to use for prediction",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print what is being annotated",
    )
    parser.add_argument(
        "-l",
        "--label",
        dest="label_id",
        type=str,
        default=None,
        help="Last part and unique identifier of the label id under which to save the newly generated labels. If empty, the reuslts will not be saved.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="Comma seperated list of dataset-names or dataset-train/valuation pairs to only annotate (e.g. test-01,bumsbiene:train,test-02:val). Leave this empty and set -tod to get the trained on dataset. If a reviewset file is given, the reviewset files will be filtered by the given dataset. If no reviewset file is given, the dataset file will be annotated.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint to use for prediction (default is last)",
    )
    parser.add_argument(
        "-tod",
        "--trained_on_dataset",
        action="store_true",
        help="Choose the trained on dataset instead of giving dataset names",
    )
    parser.add_argument(
        "-ftor",
        "--filter_trained_on_reviews",
        action="store_true",
        help="Filter out reviews that were trained on",
    )
    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name of model artifact to use (wandb run name)",
    )
    parser.add_argument(
        "reviewset_files",
        type=str,
        nargs="*",
        default=None,
        help="All reviewset files to annotate. If a dataset is given, the reviewset files will be filtered by the given dataset. If no dataset is given, all reviewset files will be annotated.",
    )

    return parser.parse_args(), parser.format_help()


def main():
    args, _ = arg_parse()

    assert (
        args.datasets or args.reviewset_files or args.trained_on_dataset
    ), "No reviewset files or datasets or trained on flag given"
    assert not (
        args.datasets and args.trained_on_dataset
    ), "Cannot specify both datasets and trained on flag"

    generation_config = utils.get_config(args.generation_config)
    generator = Generator(args.artifact_name, args.checkpoint, generation_config)

    datasets = []
    if args.datasets:
        for dataset in args.datasets.split(","):
            dataset_name, dataset_part = (
                dataset.split(":") if ":" in dataset else (dataset, None)
            )
            datasets.append((dataset_name, dataset_part))
    elif args.trained_on_dataset:
        dataset_name = utils.get_config_from_artifact(args.artifact_name)["dataset"][
            "version"
        ]
        datasets.append((dataset_name, "test"))

    if args.reviewset_files:
        reviewset = ReviewSet.from_files(*args.reviewset_files)
        filtered_reviewset = reviewset
    else:
        dataset_paths = [utils.get_dataset_path(dataset[0]) for dataset in datasets]
        print(dataset_paths)
        reviewset = ReviewSet.from_files(*dataset_paths)
        filtered_reviewset = reviewset.filter_with_label_strategy(
            DatasetSelectionStrategy(*datasets), inplace=False
        )

    if args.filter_trained_on_reviews:
        trained_on_dataset = utils.get_config_from_artifact(args.artifact_name)[
            "dataset"
        ]["version"]
        filtered_reviewset = filtered_reviewset.filter_with_label_strategy(
            DatasetSelectionStrategy((trained_on_dataset, "train")),
            invert=True,
            inplace=False,
        )

    label_id = None
    if args.label_id:
        label_id = f"model-{args.artifact_name}-{args.label_id}"

    generator.generate_label(
        filtered_reviewset, label_id=label_id, verbose=not args.quiet
    )
    if label_id:
        reviewset.save_as(input("Save as: "))


if __name__ == "__main__":
    main()
