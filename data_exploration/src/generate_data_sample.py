import random
import json
import os
from typing import Union
from pathlib import Path
import numpy as np
import argparse
from utils import get_slurm_client
import pandas as pd
from data_loading import read_data
from helpers.review import Review
from helpers.review_set import ReviewSet


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
):
    """Expects dataframes to have `product_category` column which specifies original category of entries"""

    if input_type not in ["parquet", "tsv"]:
        raise ValueError(f"input type {input_type} not supported")

    if output_type not in ["parquet", "tsv", "json"]:
        raise ValueError(f"output type {output_type} not supported")

    df = read_data(data_source, input_type)
    dummy_df = pd.DataFrame(columns=df.columns)
    assert (
        "product_category" in df.columns
    ), "`product_category` column must be present in dataframe"

    sampling_random_state = random.randint(0, 100000000)
    print(f"Using {sampling_random_state} to seed dask sampling function")

    total_row_count = df.shape[0].compute()

    sample_size = sample_size * n_samples
    # if we use this fraction from each group, there should be roughly `sample_size` many results
    fraction = sample_size / total_row_count
    print(f"Retrieving {fraction * 100}% of all data")

    all_samples_df = (
        df.groupby("product_category")
        .apply(
            lambda x: x.sample(
                frac=fraction, random_state=sampling_random_state, axis="index"
            ),
            meta=dummy_df,
        )
        .compute()
    )

    # shuffel by sampling because you can't shuffle a dataframe. TODO: our seed is not respected here
    all_samples_df = all_samples_df.sample(frac=1).reset_index(drop=True)
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
            json_v3 = {"version": ReviewSet.latest_version, "reviews": {}}
            # iterate through all rows to insert data into json
            for index, row in sample_df.iterrows():
                review_data = row.to_dict()
                review_data["labels"] = {}
                review_data = {
                    k: v
                    for k, v in review_data.items()
                    if k in list(Review.REVIEW_ATTRIBUTES)
                }
                # dump review_data dict at the review_id key
                json_v3["reviews"][row["review_id"]] = review_data
            with open(Path(output_dir, f"{base_name}.json"), "w") as file:
                json.dump(json_v3, file)

        else:
            sample_df.repartition(partition_size="100MB").to_parquet(
                output_dir,
                engine="pyarrow",
                name_function=lambda i: f"{base_name}_part{i}.parquet",
            )


def main():
    args, _ = parse_args()

    get_slurm_client(
        nodes=5,
        cores=40,
        processes=2,
        memory="512GB",
        suppress_output=args.quiet,
    )

    sample_data(
        data_source=args.input_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        n_samples=args.n_samples,
        input_type=args.input_type,
        output_type=args.output_type,
    )


if __name__ == "__main__":
    main()
