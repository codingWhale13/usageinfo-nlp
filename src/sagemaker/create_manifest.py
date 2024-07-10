#!/usr/bin/env python3

import argparse
import math
import json
import random
from src.review_set import ReviewSet
from itertools import zip_longest


def parse_args():
    def number_of_tasks(x):
        if x != "all":
            try:
                x = int(x)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"{x} is not 'all' or an integer literal"
                )

        return x

    parser = argparse.ArgumentParser(
        description="Generate a manifest file for a labelling job from a sampled file and our golden file. File format can be either tsv or json."
    )
    parser.add_argument(
        "sample_file",
        type=str,
        help="Sample reviewset file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Name of the output manifest file",
    )
    parser.add_argument(
        "--number-of-tasks",
        "-n",
        default="all",
        type=number_of_tasks,
        help="The number of tasks (MTurk HITs) in the output manifest (default is to use all available samples)",
    )
    parser.add_argument(
        "--reviews-per-task",
        "-r",
        default=10,
        type=int,
        help="The number of reviews for each task (default is 10)",
    )

    return parser.parse_args(), parser.format_help()


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def main():
    args, help_text = parse_args()

    review_set = ReviewSet.from_files(args.sample_file)

    reviews_per_task = args.reviews_per_task
    if args.number_of_tasks == "all":
        number_of_tasks = int(math.ceil(len(review_set) / reviews_per_task))
    else:
        number_of_tasks = args.number_of_tasks
        if number_of_tasks * reviews_per_task > len(review_set):
            print(
                f"WARNING: Only {len(review_set)} samples are available which is less than requested."
            )

    manifest = []
    SOURCE_COLUMN = "review_body"
    LABEL_COLUMN = "labels"

    for count, reviews_batch in enumerate(grouper(review_set, reviews_per_task)):
        if count >= number_of_tasks:
            break

        reviews_batch = list(reviews_batch)
        random.shuffle(reviews_batch)

        source = []
        metadata = []
        for review in reviews_batch:
            if review is None:
                continue
            source.append(review[SOURCE_COLUMN])

            datapoint_metadata = {}
            for column in review.data.keys():
                if column != SOURCE_COLUMN and column != LABEL_COLUMN:
                    datapoint_metadata[column] = review[column]

            datapoint_metadata["review_id"] = review.review_id
            datapoint_metadata["customUsageOptions"] = []
            datapoint_metadata["annotations"] = []

            metadata.append(datapoint_metadata)

        manifest.append({"source": json.dumps(source), "metadata": metadata})

    json_string_dataset = [json.dumps(row, ensure_ascii=False) for row in manifest]
    formatted_json_string = "\n".join(json_string_dataset)

    with open(args.output_file, "w", encoding="utf8") as file:
        file.write(formatted_json_string)


if __name__ == "__main__":
    main()
