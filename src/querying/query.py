import argparse
import os
import sys
from dotenv import load_dotenv
from dask import dataframe
from dask_sql import Context


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Do some queries on the dataset.")
    arg_parser.add_argument(
        "--data-dir",
        "-d",
        action="store",
        dest="DATA_DIR",
        default=os.getenv("DATA_DIR"),
        metavar="DATA_DIR",
        help="directory containing a number of tsv files",
    )
    arg_parser.add_argument(
        "QUERY_FILES",
        action="store",
        nargs="+",
        metavar="QUERY_FILE",
        help="text files, each containing one or more SQL queries",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def query_dataset(data_dir: str, queries: list[tuple[list[str], str]]):
    dataset = dataframe.read_csv(os.path.join(data_dir, r"*.tsv"), sep="\t")
    context = Context()
    context.create_table("reviews", dataset)

    for sub_queries, query_file in queries:
        query_file_path, _ = os.path.splitext(query_file)

        for i in range(len(sub_queries)):
            result = context.sql(sub_queries[i])
            result.compute()
            result.to_csv(
                query_file_path + f"_result_{i + 1}.tsv", sep="\t", single_file=True
            )


def main():
    load_dotenv()
    args, help_text = parse_args()

    DATA_DIR: str = args.DATA_DIR
    QUERY_FILES: str = args.QUERY_FILES

    if DATA_DIR is None:
        print("did not find any data directory, add it into your .env file or specify it with -d", file=sys.stderr)
        print(help_text)
        exit(1)

    if not os.path.isdir(DATA_DIR):
        print("data directory does not exist", file=sys.stderr)
        print(help_text)
        exit(2)

    queries = []
    for file_path in QUERY_FILES:
        if os.path.isfile(file_path):
            with open(file_path, "r") as query_file:
                sub_queries = query_file.read().split(";")
            sub_queries = list(filter(None, sub_queries))
            sub_queries = [sub_query.replace("\n", "") for sub_query in sub_queries]

            if sub_queries:
                queries.append((sub_queries, file_path))

    if not queries:
        print("found no valid query file", file=sys.stderr)
        print(help_text)
        exit(3)

    query_dataset(DATA_DIR, queries)


if __name__ == "__main__":
    main()
