import argparse
import pandas as pd
import os
import yaml
from typing import List

from extract_reviews import extract_reviews_with_usage_options_from_json

DEFAULT_PATH = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/datasets"
DATASETS_DIR = os.getenv("DATASETS", default=DEFAULT_PATH)


def arg_parse():
    parser = argparse.ArgumentParser(description="Create a new training dataset.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of dataset version",
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
        help="Percentage of the data to be used for testing",
    )
    return parser.parse_args(), parser.format_help()


def create_dataset_dir(name: str):
    dataset_dir = os.path.join(DATASETS_DIR, name)
    if os.path.exists(dataset_dir):
        if input("Dataset already exists. Overwrite? (y/N): ").lower() != "y":
            exit("Aborted.")
    else:
        os.mkdir(dataset_dir)

    return dataset_dir


def create_dataset(files: List[str], test_split: float):
    reviews = [extract_reviews_with_usage_options_from_json(file) for file in files]

    df = pd.concat(reviews, axis=0, ignore_index=True)

    df_train = df.sample(frac=1 - test_split, random_state=42)
    df_test = df.drop(df_train.index)

    return df_train, df_test


def create_yml(dataset_version, test_split, files, dataset_dir):
    dict_args = {
        "version": dataset_version,
        "test_split": test_split,
        "files": files,
    }
    with open(os.path.join(dataset_dir, "config.yml"), "w") as file:
        yaml.dump(dict_args, file)


def main():
    args, _ = arg_parse()
    files = args.files
    dataset_version = args.dataset_name
    test_split = args.test_split

    dataset_dir = create_dataset_dir(name=dataset_version)
    df_train, df_test = create_dataset(
        dataset_dir=dataset_dir, files=files, test_split=test_split
    )

    df_train.to_json(os.path.join(dataset_dir, "train_data.json"))
    df_test.to_json(os.path.join(dataset_dir, "test_data.json"))

    create_yml(
        dataset_version=dataset_version,
        test_split=test_split,
        files=files,
        dataset_dir=dataset_dir,
    )


if __name__ == "__main__":
    main()
