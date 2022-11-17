import random
import os
from typing import Union
from pathlib import Path
import argparse

from utils import get_slurm_client
from data_loading import read_data, review_data_dtypes


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
        choices=["parquet", "tsv"],
        default="parquet",
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
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
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
    input_type="parquet",
    output_type="tsv",
):
    """Expects dataframes to have `product_category` column which specifies original category of entries"""

    if input_type not in ["parquet", "tsv"]:
        raise ValueError(f"input type {input_type} not supported")

    if output_type not in ["parquet", "tsv"]:
        raise ValueError(f"output type {output_type} not supported")

    df = read_data(data_source, input_type)

    assert (
        "product_category" in df.columns
    ), "`product_category` column must be present in dataframe"

    sampling_random_state = random.randint(0, 100000000)
    print(f"Using {sampling_random_state} to seed dask sampling function")

    total_row_count = df.shape[0].compute()

    # if we use this fraction from each group, there should be roughly `sample_size` many results
    fraction = sample_size / total_row_count
    print(f"Retrieving {fraction * 100}% of all data")

    sample_df = df.groupby("product_category").apply(
        lambda x: x.sample(
            frac=fraction, random_state=sampling_random_state, axis="index"
        ),
        meta=review_data_dtypes,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = f"sample_{sampling_random_state}"

    if output_type == "tsv":
        return sample_df.compute().to_csv(
            Path(output_dir, f"{base_name}.tsv"),
            index=False,
            mode="w",
            sep="\t",
        )

    return sample_df.repartition(partition_size="100MB").to_parquet(
        output_dir,
        engine="pyarrow",
        name_function=lambda i: f"{base_name}_part{i}.parquet",
    )


def main():
    args, help_text = parse_args()

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
        input_type=args.input_type,
        output_type=args.output_type,
    )


if __name__ == "__main__":
    main()
