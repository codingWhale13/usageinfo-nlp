import argparse
import os
import sys
from dotenv import load_dotenv
from dask_sql import Context
import sqlparse

from utils import get_slurm_client
from data_loading import read_data


review_data_dtypes = {
    "marketplace": "string",
    "customer_id": "string",
    "review_id": "string",
    "product_id": "string",
    "product_parent": "string",
    "product_title": "string",
    "product_category": "string",
    "star_rating": "int64",
    "helpful_votes": "int64",
    "total_votes": "int64",
    "vine": "string",
    "verified_purchase": "string",
    "review_headline": "string",
    "review_body": "string",
    "review_date": "string",
}


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Do some queries on the dataset.")
    arg_parser.add_argument(
        "--data-dir",
        "-d",
        action="store",
        dest="DATA_DIR",
        default=os.getenv("DATA_DIR"),
        metavar="DATA_DIR",
        help="directory containing a number of parquet files",
    )
    arg_parser.add_argument(
        "--output-to-file",
        "-f",
        action="store_true",
        dest="OUTPUT_TO_FILE",
        help="output the result of the queries to individual files instead of stdout",
    )
    arg_parser.add_argument(
        "QUERY_FILES",
        action="store",
        nargs="+",
        metavar="QUERY_FILE",
        help="text files, each containing one or more SQL queries",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def query_dataset(
    data_dir: str, queries: list[tuple[list[str], str]], output_to_file: bool = False
):
    dataset = read_data(data_dir, "parquet").astype(review_data_dtypes)
    context = Context()
    context.create_table("reviews", dataset)

    for sub_queries, query_file in queries:
        query_file_path, _ = os.path.splitext(query_file)

        for i in range(len(sub_queries)):
            result = context.sql(sub_queries[i])
            if output_to_file:
                result.to_csv(
                    f"{query_file_path}_result_{i + 1}.tsv", sep="\t", single_file=True
                )
            else:
                print(
                    f"{query_file} result {i + 1}:",
                    result.head(10),
                    "\n----------------------------\n",
                    sep="\n",
                )


def main():
    load_dotenv()
    args, help_text = parse_args()

    DATA_DIR: str = args.DATA_DIR
    QUERY_FILES: str = args.QUERY_FILES
    OUTPUT_TO_FILE: bool = args.OUTPUT_TO_FILE

    if DATA_DIR is None:
        print(
            "did not find any data directory, add it into your .env file or specify it with -d",
            file=sys.stderr,
        )
        print(help_text)
        exit(1)

    if not os.path.isdir(DATA_DIR):
        print("data directory does not exist", file=sys.stderr)
        print(help_text)
        exit(2)

    get_slurm_client(
        nodes=6,
        cores=128,
        processes=8,
        memory="512GB",
    )

    queries = []
    for file_path in QUERY_FILES:
        if os.path.isfile(file_path):
            with open(file_path, "r") as query_file:
                sub_queries = list(map(str, sqlparse.parse(query_file.read())))

            if sub_queries:
                queries.append(
                    (sub_queries, file_path)
                )  # safe file path with queries for naming of output files

    if not queries:
        print("found no valid query file", file=sys.stderr)
        print(help_text)
        exit(3)

    query_dataset(DATA_DIR, queries, OUTPUT_TO_FILE)


if __name__ == "__main__":
    main()
