import argparse
import pandas as pd
import os
import json
import yaml
import random
from typing import List
import glob

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
        help="Percentage of the data to be used for testing (default. 0.1)",
    )
    parser.add_argument(
        "-u",
        "--usage-split",
        type=float,
        default=None,
        help="Percentage of the data containing usage options (default: original)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed to shuffle data with",
    )
    return parser.parse_args(), parser.format_help()


def get_all_files(paths: List[str]):
    files = []
    for path in paths:
        files.extend(glob.glob(path))
    return files


def create_dataset_dir(name: str):
    dataset_dir = os.path.join(DATASETS_DIR, name)
    if os.path.exists(dataset_dir):
        if input("Dataset already exists. Overwrite? (y/N): ").lower() != "y":
            exit("Aborted.")
    else:
        os.mkdir(dataset_dir)

    return dataset_dir


def split_usageinfo_reviews(reviews):
    reviews_with_usage, reviews_without_usage = [], []
    for review in reviews:
        label = review["label"]
        if label["annotations"] or label["customUsageOptions"]:
            reviews_with_usage.append(review)
        else:
            reviews_without_usage.append(review)
    return reviews_with_usage, reviews_without_usage


def filter_merge_reviews(list1, list2, split):
    target_number = round((len(list1) / split) * (1 - split))
    random.shuffle(list2)
    return list1 + list2[:target_number]


def create_dataset(files: List[str], usage_split, test_split: float):
    train_data = {"reviews": [], "maxReviewIndex": 0}
    test_data = {"reviews": [], "maxReviewIndex": 0}
    result = list()
    for file in files:
        with open(file, "r") as f:
            result.extend(json.load(f)["reviews"])

    reviews_with_usage, reviews_without_usage = split_usageinfo_reviews(result)

    result = (
        filter_merge_reviews(reviews_with_usage, reviews_without_usage, usage_split)
        if len(reviews_with_usage) / len(result) <= usage_split
        else filter_merge_reviews(
            reviews_without_usage, reviews_with_usage, 1 - usage_split
        )
    )

    random.shuffle(result)
    split_index = int(len(result) * (1 - test_split))
    train_data["reviews"] = result[:split_index]
    test_data["reviews"] = result[split_index:]

    return train_data, test_data


def create_yml(dataset_version, usage_split, test_split, files, dataset_dir):
    dict_args = {
        "version": dataset_version,
        "usage_split": usage_split if usage_split else "original",
        "test_split": test_split,
        "files": files,
    }
    with open(os.path.join(dataset_dir, "config.yml"), "w") as file:
        yaml.dump(dict_args, file)


def main():
    args, _ = arg_parse()
    files = get_all_files(args.files)
    dataset_version = args.dataset_name
    usage_split = args.usage_split
    test_split = args.test_split
    seed = args.seed

    random.seed(seed)

    dataset_dir = create_dataset_dir(name=dataset_version)
    train_data, test_data = create_dataset(
        files=files, usage_split=usage_split, test_split=test_split
    )
    with open(os.path.join(dataset_dir, "train_data.json"), "w") as output_file:
        json.dump(train_data, output_file)

    with open(os.path.join(dataset_dir, "test_data.json"), "w") as output_file:
        json.dump(test_data, output_file)

    create_yml(
        dataset_version=dataset_version,
        usage_split=usage_split,
        test_split=test_split,
        files=files,
        dataset_dir=dataset_dir,
    )


if __name__ == "__main__":
    main()
