import os
from pathlib import Path
from typing import Union

import pandas as pd
import json
import wget
import dask.dataframe as dd

from bs4 import BeautifulSoup
import pandas as pd
import textstat
import dask.dataframe as dd
import html


# our data processing pipeline from the raw data is: read_data, pre_process, filter, add_metadata, write_data

review_data_dtypes = {
    "marketplace": "category",
    "customer_id": "string[pyarrow]",
    "review_id": "string[pyarrow]",
    "product_id": "string[pyarrow]",
    "product_parent": "string[pyarrow]",
    "product_title": "string[pyarrow]",
    "product_category": "string[pyarrow]",
    "star_rating": "int8",
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


def read_data_from_tsv(path: Union[Path, str], tsv_pre_processed=False) -> dd.DataFrame:
    """Reads data from tsv files
    requires specific columns to be present"""
    if os.path.isdir(path):
        path = os.path.join(path, "*.tsv")

    review_data_dtypes_copy = review_data_dtypes.copy()

    if tsv_pre_processed:
        review_data_dtypes_copy["vine"] = "int8"
        review_data_dtypes_copy["verified_purchase"] = "int8"

    df = dd.read_csv(
        path,
        sep="\t",
        on_bad_lines="warn",
        dtype=review_data_dtypes_copy,
        header=0,
        quoting=3,
    )

    return df


def read_data(
    path: Union[Path, str], file_type: str = "parquet", tsv_pre_processed: bool = False
) -> dd.DataFrame:
    if file_type == "parquet":
        return dd.read_parquet(path, engine="pyarrow")
    if file_type == "tsv":
        return read_data_from_tsv(path, tsv_pre_processed=tsv_pre_processed)
    raise ValueError(f"File type {file_type} not supported")


def convert_tsv_to_parquet(
    tsv_dir: Union[Path, str], parquet_dir: Union[Path, str], tsv_pre_processed=False
):
    """Converts tsv files to parquet files
    requires specific columns to be present"""
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)

    df = read_data_from_tsv(tsv_dir, tsv_pre_processed=tsv_pre_processed)

    write_data(df, parquet_dir, file_type="parquet")


def write_data(df: dd.DataFrame, path: Union[Path, str], file_type: str = "parquet"):
    if not os.path.exists(path):
        os.makedirs(path)

    if file_type == "parquet":
        return df.repartition(partition_size="100MB").to_parquet(
            path, engine="pyarrow", name_function=lambda i: f"data_{i}.parquet"
        )
    if file_type == "tsv":
        return df.to_csv(path + "/data__*.tsv", index=False, mode="w", sep="\t")
    raise ValueError(f"File type {file_type} not supported")


def filter_out_categories(files: list, categories: list) -> list:
    good_files = []
    for file in files:
        category = file.split("/")[-1].split(".")[0].split("_us_")[-1].split("_v")[0]
        if category in categories:
            good_files.append(file)

    return good_files


def limit_word_count(df: dd.DataFrame, max_words: int = 400) -> dd.DataFrame:
    df["review_body"] = df["review_body"].apply(
        lambda x: " ".join(x.split()[:max_words]),
        meta=("review_body", "string[pyarrow]"),
    )
    return df


def parse_review_bodies(df: dd.DataFrame) -> dd.DataFrame:
    df["review_body_original"] = df["review_body"]
    df["review_body"] = df["review_body"].apply(
        lambda x: " ".join(
            BeautifulSoup(html.unescape(x), "html.parser")
            .get_text(separator=" ")
            .strip()
            .encode("utf-8", "ignore")
            .decode()
            .split()
        )
        if not pd.isna(x)
        else "",
        meta=("review_body", "string[pyarrow]"),
    )
    return df


def filter_reviews_by_categories(df: dd.DataFrame, categories: list) -> dd.DataFrame:
    return df[df["product_category"].isin(categories)]


def filter_reviews_by_word_count(
    df: dd.DataFrame, min_word_count: int = 0, max_word_count: int = float("inf")
) -> dd.DataFrame:
    df = df.dropna(subset=["review_body"])
    df = df[df["review_body"].str.split().str.len() > min_word_count]
    df = df[df["review_body"].str.split().str.len() < max_word_count]
    return df


def filter_reviews_from_potential_bots(
    df: dd.DataFrame, max_reviews_per_day_per_customer: int = 2
) -> dd.DataFrame:
    df2 = (
        df.groupby(["review_date", "customer_id"])
        .review_id.count()
        .rename("review_count")
        .persist()
    )
    df2 = df2[df2 > max_reviews_per_day_per_customer].persist()
    df2 = (
        df2.groupby("customer_id").max().rename("max_review_count").to_frame().persist()
    )
    return df[~df.customer_id.isin(df2.index.compute())]


def pre_process(df: dd.DataFrame) -> dd.DataFrame:
    df["vine"] = df["vine"].replace({"N": 0, "Y": 1}).astype("int8")
    df["verified_purchase"] = (
        df["verified_purchase"].replace({"N": 0, "Y": 1}).astype("int8")
    )
    df = parse_review_bodies(df)

    return df


def filter(df: dd.DataFrame) -> dd.DataFrame:
    with open("good_to_label_categories.txt", "r") as file:
        categories = [category.strip() for category in file.readlines()]

    df = filter_reviews_from_potential_bots(df, max_reviews_per_day_per_customer=30)
    df = filter_reviews_by_categories(df, categories)
    df = filter_reviews_by_word_count(df, min_word_count=4)
    df = limit_word_count(df, max_words=400)

    df = df[(df["verified_purchase"] == 1) | (df["vine"] == 1)]

    return df


def add_metadata(df: dd.DataFrame) -> dd.DataFrame:
    usage_indicators = [
        "use",
        "mostly for",
        "good for",
        "suitable for",
        "perfect for",
        "ideal for",
        "works well for",
        "designed for",
        "intended for",
        "applications include",
        "multi-purpose",
        "purpose",
        "great for",
        "utilize",
        "utilise",
        "helpful",
        "using",
        "excellent for",
    ]

    df["review_body_word_count"] = df["review_body"].apply(
        lambda x: textstat.lexicon_count(x, removepunct=False),
        meta=("review_body_word_count", "int64"),
    )
    df["review_body_sent_count"] = df["review_body"].apply(
        lambda x: textstat.sentence_count(x), meta=("review_body_sent_count", "int64")
    )
    df["review_body_char_count"] = df["review_body"].apply(
        lambda x: textstat.char_count(x), meta=("review_body_char_count", "int64")
    )
    df["review_body_lett_count"] = df["review_body"].apply(
        lambda x: textstat.letter_count(x), meta=("review_body_lett_count", "int64")
    )

    mask = df["review_body_word_count"] != 0
    df["review_body_avg_word_length"] = (
        df["review_body_lett_count"] / df["review_body_word_count"]
    )
    df["review_body_avg_word_length"] = df["review_body_avg_word_length"].mask(~mask, 0)

    df["review_body_reading_time"] = df["review_body"].apply(
        lambda x: textstat.reading_time(x), meta=("review_body_reading_time", "float64")
    )

    df["review_body_usage_count"] = df["review_body"].apply(
        lambda x: sum([x.lower().count(word) for word in usage_indicators]),
        meta=("review_body_usage_count", "int64"),
    )
    df["review_body_usage_density"] = (
        df["review_body_usage_count"] / df["review_body_word_count"]
    )
    df["review_body_usage_density"] = df["review_body_usage_density"].mask(~mask, 0)

    df["review_body_flesch_complexity"] = df["review_body"].apply(
        lambda x: textstat.flesch_reading_ease(str(x)),
        meta=("review_body_flesch_complexity", "float64"),
    )

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
