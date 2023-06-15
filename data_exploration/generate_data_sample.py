import random
import json
import os
from typing import Union
from pathlib import Path
import numpy as np
import argparse
import pandas as pd
from data_loading import read_data
from helpers.review import Review
from helpers.review_set import ReviewSet
from dask.diagnostics import ProgressBar
import dask as dd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a sample of the data.")
    parser.add_argument(
        "-i",
        "--input-type",
        action="store",
        dest="input_type",
        choices=["parquet", "tsv"],
        default="parquet",
        help="input file type",
    )
    parser.add_argument(
        "-o",
        "--output-type",
        action="store",
        dest="output_type",
        choices=["parquet", "tsv", "json"],
        default="json",
        help="output type of the sample, for parquet the sample will be split up into multiple files",
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        action="store",
        dest="sample_size",
        default=1000,
        type=int,
        help="size of the sample",
    )
    parser.add_argument(
        "-n",
        "--number-of-samples",
        action="store",
        dest="n_samples",
        default=1,
        type=int,
        help="Number of samples",
    )
    parser.add_argument(
        "-r",
        "--star-rating",
        action="store_true",
        default=False,
        help="Sample star_rating equally",
    )
    parser.add_argument(
        "-c",
        "--product-category",
        action="store_true",
        default=False,
        help="Sample product category equally",
    ),

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        default=True,
        help="suppress SLURM output",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        action="store",
        dest="output_dir",
        default=Path.cwd(),
        help="directory to output the sample to",
    )
    parser.add_argument(
        "input_dir",
        action="store",
        help="directory containing the data to sample",
    )

    return parser.parse_args(), parser.format_help()


def sample_data(
    data_source: Union[Path, str],
    output_dir: Union[Path, str],
    sample_size: int = 1000,
    n_samples: int = 1,
    input_type="parquet",
    output_type="tsv",
    sample_star_rating_equally=False,
    sample_product_category_equally=False,
):
    """Expects dataframes to have `product_category` column which specifies original category of entries"""

    if input_type not in ["parquet", "tsv"]:
        raise ValueError(f"input type {input_type} not supported")

    if output_type not in ["parquet", "tsv", "json"]:
        raise ValueError(f"output type {output_type} not supported")

    print("Reading data")
    df = read_data(data_source, input_type)
    dummy_df = pd.DataFrame(columns=df.columns)
    assert (
        "product_category" in df.columns
    ), "`product_category` column must be present in dataframe"

    sampling_random_state = random.randint(0, 100000000)
    print(f"Using {sampling_random_state} to seed dask sampling function")

    PRODUCT_CATEGORY_COUNT = 28
    STAR_RATING_COUNT = 5

    def caluclate_sample_size(
        total_df: dd.dataframe,
        product_category_df: dd.dataframe,
        star_rating_df: dd.dataframe,
        total_sample_size,
    ):
        if sample_product_category_equally and sample_star_rating_equally:
            return int(total_sample_size / (PRODUCT_CATEGORY_COUNT * STAR_RATING_COUNT))
        elif sample_product_category_equally and not sample_star_rating_equally:
            return int(
                (total_sample_size / PRODUCT_CATEGORY_COUNT)
                * (len(star_rating_df) / len(product_category_df))
            )
        elif not sample_product_category_equally and sample_star_rating_equally:
            return int(
                (total_sample_size / STAR_RATING_COUNT)
                * (len(product_category_df) / len(total_df))
            )
        else:
            return int(
                (
                    total_sample_size
                    * (len(star_rating_df) / len(product_category_df))
                    * (len(product_category_df) / len(total_df))
                )
            )

    all_samples_df = df.groupby("product_category").apply(
        lambda x: x.groupby("star_rating").apply(
            lambda x_2: x_2.sample(
                n=caluclate_sample_size(df, x, x_2, sample_size * n_samples),
                random_state=sampling_random_state,
                axis="index",
            )
        ),
        meta=dummy_df,
    )

    print("Starting sampling")
    with ProgressBar():
        all_samples_df = all_samples_df.compute().dropna()

    # shuffle by sampling because you can't shuffle a dataframe
    all_samples_df = all_samples_df.sample(
        frac=1, random_state=sampling_random_state
    ).reset_index(drop=True)
    samples = np.array_split(all_samples_df, n_samples)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(samples)):
        base_name = f"sample_{sampling_random_state}-{i}"
        sample_df = samples[i]

        if output_type == "tsv":
            sample_df.to_csv(
                Path(output_dir, f"{base_name}.tsv"),
                index=False,
                mode="w",
                sep="\t",
            )
        elif output_type == "json":
            # Create empty json to insert Reviews
            json_review_set = {"version": ReviewSet.latest_version, "reviews": {}}
            # iterate through all rows to insert data into json
            for index, row in sample_df.iterrows():
                review_data = row.to_dict()
                review_data["labels"] = {}
                review_data["augmentations"] = []
                review_data = {
                    k: v
                    for k, v in review_data.items()
                    if k in list(Review.review_attributes)
                }
                # dump review_data dict at the review_id key
                json_review_set["reviews"][row["review_id"]] = review_data
            with open(Path(output_dir, f"{base_name}.json"), "w") as file:
                json.dump(json_review_set, file)

        else:
            sample_df.repartition(partition_size="100MB").to_parquet(
                output_dir,
                engine="pyarrow",
                name_function=lambda i: f"{base_name}_part{i}.parquet",
            )


def main():
    args, _ = parse_args()

    sample_data(
        data_source=args.input_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        n_samples=args.n_samples,
        input_type=args.input_type,
        output_type=args.output_type,
        sample_product_category_equally=args.product_category,
        sample_star_rating_equally=args.star_rating,
    )


if __name__ == "__main__":
    main()
