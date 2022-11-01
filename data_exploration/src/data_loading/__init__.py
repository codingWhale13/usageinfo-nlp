import os
from pathlib import Path, PurePath
from typing import Union

import wget
import pandas as pd
import pyarrow


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


def read_data(file_path: Union[Path, str]) -> pd.DataFrame:

    dtype={
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


    df = pd.read_csv(
        file_path,
        sep="\t",
        on_bad_lines="warn",
        dtype=dtype,
        header=0,
        quoting=3
    )

    return df
