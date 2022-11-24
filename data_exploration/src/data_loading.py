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


def extract_labelled_reviews_from_json(
    path: Union[Path, str],
    extraction_func: callable,
    label_coloumn_name: string = "label",
) -> pd.DataFrame:
    with open(path, "r") as file:
        data = json.load(file)
        df = pd.DataFrame(data["reviews"])
        df[label_coloumn_name] = df["label"].apply(extraction_func)
        if label_coloumn_name != "label":
            df.drop("label", axis=1, inplace=True)
    return df


def extract_reviews_with_usage_options_from_json(
    path: Union[Path, str]
) -> pd.DataFrame:
    extract_usage_options_list = lambda x: x["customUsageOptions"] + [
        " ".join(annotation["tokens"]) for annotation in x["annotations"]
    ]
    return extract_labelled_reviews_from_json(
        path, extract_usage_options_list, "usage_options"
    )
