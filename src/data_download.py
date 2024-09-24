"""
    Load data from the raw data files and preprocess it.
"""

from pathlib import Path
from typing import Union

from bs4 import BeautifulSoup
import html
from datasets import load_dataset, Dataset
import argparse

from src.review_set import ReviewSet


GOOD_TO_LABEL_CATEGORIES = [
    "Camera",
    "Major Appliances",
    "Luggage",
    "Office Products",
    "Wireless",
    "Mobile_Electronics",
    "Baby",
    "Jewelry",
    "Lawn and Garden",
    "PC",
    "Tools",
    "Sports",
    "Grocery",
    "Home Entertainment",
    "Automotive",
    "Shoes",
    "Home Improvement",
    "Watches",
    "Toys",
    "Furniture",
    "Outdoors",
    "Home",
    "Electronics",
    "Beauty",
    "Kitchen",
    "Apparel",
    "Pet Products",
    "Musical Instruments",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and preprocess the Amazon reviews dataset."
    )
    parser.add_argument("data_path", type=str, help="Path to the raw data files.")
    parser.add_argument("out_path", type=str, help="Path to save the processed data.")
    parser.add_argument(
        "--num_proc", type=int, default=8, help="Number of processes to use."
    )
    return parser.parse_args()


def create_dataset(path, out_path, num_proc=8):
    """
    We asume the data is already downloaded and lies in the .tsv format in the specified folder.
    """
    data = read_data_from_tsv(path, num_proc)
    pre_processed_data = pre_process(data, num_proc)
    filtered_data = filter(pre_processed_data, num_proc)

    review_dict = filtered_data.to_dict()
    list_of_review_dicts = [
        dict(zip(review_dict, t)) for t in zip(*review_dict.values())
    ]
    review_dict = {}
    for item in list_of_review_dicts:
        review_id = item.pop("review_id")
        item["labels"] = {}
        review_dict[review_id] = item
    review_set = ReviewSet.from_dict({"version": 5, "reviews": review_dict})
    review_set.save(out_path)


def read_data_from_tsv(path: Union[Path, str], num_proc: int) -> Dataset:
    """Reads data from tsv files
    requires specific columns to be present"""

    dataset = load_dataset(
        "csv",
        split="train",
        data_dir=path,
        num_proc=num_proc,
        delimiter="\t",
    )

    return dataset


def pre_process(dataset: Dataset, num_proc: int) -> Dataset:
    def pre_process(examples):
        examples["vine"] = examples["vine"].replace("N", "0")
        examples["vine"] = examples["vine"].replace("Y", "1")
        examples["vine"] = int(examples["vine"])

        examples["verified_purchase"] = examples["verified_purchase"].replace("N", "0")
        examples["verified_purchase"] = examples["verified_purchase"].replace("Y", "1")
        examples["verified_purchase"] = int(examples["verified_purchase"])
        examples = parse_review_bodies(examples)
        return examples

    return dataset.map(pre_process, desc="Preprocessing data", num_proc=num_proc)


def parse_review_bodies(dataset: Dataset) -> Dataset:
    dataset["review_body"] = " ".join(
        BeautifulSoup(html.unescape(dataset["review_body"]), "html.parser")
        .get_text(separator=" ")
        .strip()
        .encode("utf-8", "ignore")
        .decode()
        .split()
    )
    return dataset


def filter(dataset: Dataset, num_proc: int) -> Dataset:
    dataset = filter_reviews_from_potential_bots(
        dataset, num_proc, max_reviews_per_day_per_customer=30
    )
    dataset = filter_reviews_by_categories(dataset, num_proc)
    dataset = filter_reviews_by_word_count(dataset, num_proc, min_word_count=4)
    dataset = limit_word_count(dataset, num_proc, max_words=400)

    dataset = dataset.filter(
        lambda x: x["verified_purchase"] == 1 or x["vine"] == 1, num_proc=num_proc
    )

    return dataset


def filter_reviews_from_potential_bots(
    dataset: Dataset, num_proc: int, max_reviews_per_day_per_customer: int = 2
) -> Dataset:
    dataset2 = dataset.to_pandas().groupby(["customer_id", "review_date"]).size()
    dataset2 = dataset2[dataset2 > max_reviews_per_day_per_customer]
    dataset2 = (
        dataset2.groupby("customer_id").max().rename("max_review_count").to_frame()
    )
    if dataset2.empty:
        return dataset
    return dataset.filter(
        lambda x: x["customer_id"] not in dataset2["customer_id"].values,
        num_proc=num_proc,
    )


def filter_reviews_by_categories(dataset: Dataset, num_proc: int) -> Dataset:
    return dataset.filter(
        lambda x: x["product_category"] in GOOD_TO_LABEL_CATEGORIES, num_proc=num_proc
    )


def filter_reviews_by_word_count(
    dataset: Dataset,
    num_proc: int,
    min_word_count: int = 0,
    max_word_count: int = float("inf"),
) -> Dataset:
    dataset = dataset.filter(
        lambda x: len(x["review_body"].split()) > min_word_count, num_proc=num_proc
    )
    return dataset.filter(
        lambda x: len(x["review_body"].split()) < max_word_count, num_proc=num_proc
    )


def limit_word_count(dataset: Dataset, num_proc: int, max_words: int = 400) -> Dataset:
    def truncate_review_body(examples):
        examples["review_body"] = " ".join(examples["review_body"].split()[:max_words])
        return examples

    return dataset.map(
        truncate_review_body, desc="Truncating review bodies", num_proc=num_proc
    )


def filter_out_categories(files: list) -> list:
    good_files = []
    for file in files:
        category = file.split("/")[-1].split(".")[0].split("_us_")[-1].split("_v")[0]
        if category in GOOD_TO_LABEL_CATEGORIES:
            good_files.append(file)

    return good_files


def extract_json_from_manifest(
    input_path: Union[Path, str], output_path: Union[Path, str] = None
):
    with open(input_path, "r") as turker_labels:
        reviews = {"reviews": [], "maxReviewIndex": 0}
        run_name = get_run_name(turker_labels)
        if output_path is None:
            output_path = f"{run_name}-output.json"
        for line in turker_labels:
            data = json.loads(line)
            review_bodies = json.loads(data["source"])
            number_of_workers_per_hit = len(data[run_name]["annotationsFromAllWorkers"])
            number_of_reviews_per_hit = len(review_bodies)
            for worker in range(number_of_workers_per_hit):
                for review in range(number_of_reviews_per_hit):
                    inner_annotations = json.loads(
                        data[run_name]["annotationsFromAllWorkers"][worker][
                            "annotationData"
                        ]["content"]
                    )
                    annotations = json.loads(inner_annotations["annotations"])[
                        "annotations"
                    ]
                    customUsageOptions = json.loads(inner_annotations["annotations"])[
                        "customUsageOptions"
                    ]
                    reviews["reviews"].append(
                        data["metadata"][review]
                        | {
                            "review_body": review_bodies[review],
                            "label": {
                                "isFlagged": False,
                                "annotations": annotations[review],
                                "customUsageOptions": customUsageOptions[review],
                            },
                            "inspectionTime": None,
                        }
                    )
        with open(output_path, "w") as output_file:
            json.dump(reviews, output_file)


def get_run_name(turker_labels):
    data = json.loads(turker_labels.readline())
    for key, value in data.items():
        if isinstance(value, dict) and "job-name" in value:
            return data[key]["job-name"]


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args.data_path, args.out_path, args.num_proc)
