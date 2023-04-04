import argparse

from training.src.generator import Generator
from training.src import utils
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
        default="generation_config",
        help="Generation config to use for prediction",
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
        help="Comma seperated list of dataset-names or dataset-train/valuation pairs to only annotate (e.g. test-01,bumsbiene:train,test-02:val)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint to use for prediction (default is last)",
    )
    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name of model artifact to use (wandb run name)",
    )
    parser.add_argument(
        "reviewset_files",
        type=str,
        nargs="+",
        help="All reviewset files to annotate",
    )

    return parser.parse_args(), parser.format_help()


def main():
    args, _ = arg_parse()

    generation_config = utils.get_config(args.generation_config)
    generator = Generator(args.artifact_name, args.checkpoint, generation_config)

    reviewset = ReviewSet.from_files(*args.reviewset_files)
    filtered_reviewset = reviewset
    if args.datasets is not None:
        datasets = []
        for dataset in args.datasets.split(","):
            dataset_name, dataset_part = (
                dataset.split(":") if ":" in dataset else (dataset, None)
            )
            datasets.append((dataset_name, dataset_part))
        filtered_reviewset = reviewset.filter_with_label_strategy(
            DatasetSelectionStrategy(datasets)
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
