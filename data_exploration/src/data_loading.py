import os
from pathlib import Path
from typing import Union

import pandas as pd
import json
import wget
import dask.dataframe as dd

review_data_dtypes = {
    "marketplace": "category",
    "customer_id": "string[pyarrow]",
    "review_id": "string[pyarrow]",
    "product_id": "string[pyarrow]",
    "product_parent": "string[pyarrow]",
    "product_title": "string[pyarrow]",
    "product_category": "string[pyarrow]",
    "star_rating": "category",
    "helpful_votes": "int64",
    "total_votes": "int64",
    "vine": "category",
    "verified_purchase": "category",
    "review_headline": "string[pyarrow]",
    "review_body": "string[pyarrow]",
    "review_date": "string[pyarrow]",
}


def fetch_files_from_txt(url_file: Union[Path, str], target_dir: Union[Path, str]):
    with open(url_file, "r") as urls:
        target_dir = os.path.expanduser(target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        file_paths = []
        for url in [url.strip() for url in urls.readlines()]:
            file_name = os.path.split(url)[1]
            file_path = os.path.join(target_dir, file_name)
            file_paths.append(file_path)

            if os.path.exists(file_path):
                print(f"Skipping {file_name}... file already exists")
                continue

            print(f"Downloading {url}...")
            wget.download(url, target_dir)
            print()

        return file_paths


def read_data_from_tsv(path: Union[Path, str]) -> dd.DataFrame:
    """Reads data from tsv files
    requires specific columns to be present"""
    if os.path.isdir(path):
        path = os.path.join(path, "*.tsv")

    df = dd.read_csv(
        path,
        sep="\t",
        on_bad_lines="warn",
        dtype=review_data_dtypes,
        header=0,
        quoting=3,
    )

    return df


def read_data(path: Union[Path, str], file_type: str = "parquet") -> dd.DataFrame:
    if file_type == "parquet":
        return dd.read_parquet(path, engine="pyarrow")
    if file_type == "tsv":
        return read_data_from_tsv(path)
    raise ValueError(f"File type {file_type} not supported")


def limit_word_count(string: str, max_words: int = 400) -> str:
    return " ".join(string.split(" ")[:max_words])


def convert_tsv_to_parquet(
    start_dir: Union[Path, str], target_dir: Union[Path, str], filter: bool = False
):
    """Converts tsv files to parquet files. If filter is True, only reviews with more than 4 words are kept, reviews are cut of at 400 words and reviews from \'bad\' categories are removed"""

    if filter:
        with open("good_to_label_categories.txt", "r") as file:
            categories = [
                category.strip().replace(" ", "_") for category in file.readlines()
            ]

        files = []
        file_names = os.listdir(start_dir)
        for file in file_names:
            if not file.startswith(start_dir):
                files.append(os.path.join(start_dir, file))

        files = filter_out_categories(files, categories)

        df = dd.read_csv(
            files,
            sep="\t",
            on_bad_lines="warn",
            dtype=review_data_dtypes,
            header=0,
            quoting=3,
            converters={"review_body": limit_word_count},
        )

        df = filter_reviews_by_word_count(df, min_word_count=4)

    else:
        df = read_data_from_tsv(start_dir)

    df.repartition(partition_size="100MB").to_parquet(target_dir, engine="pyarrow")


def filter_out_categories(files: list, categories: list) -> list:
    good_files = []
    for file in files:
        category = file.split("/")[-1].split(".")[0].split("_us_")[-1].split("_v")[0]
        if category in categories:
            good_files.append(file)

    return good_files


def filter_reviews_by_word_count(
    df: dd.DataFrame, min_word_count: int = 0, max_word_count: int = float("inf")
) -> dd.DataFrame:
    df = df.dropna(subset=["review_body"])
    df = df[df["review_body"].str.split(" ").str.len() > min_word_count]
    df = df[df["review_body"].str.split(" ").str.len() < max_word_count]
    return df


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
