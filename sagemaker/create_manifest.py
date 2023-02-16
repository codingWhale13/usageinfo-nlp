import json
import argparse
import pandas as pd
import random
from itertools import zip_longest
from ..openai.openai_pre_annotion import pre_label_format_manifest
from reproducable_context import ReproducableContext


def load_from_json(file_name):
    json_raw = json.load(open(file_name))
    df_temp = pd.DataFrame(json_raw["reviews"])
    return json.loads(df_temp.to_json(orient="records"))


def load_from_tsv(file_name):
    return json.loads(
        pd.read_csv(file_name, sep="\t", quoting=3).to_json(orient="records")
    )


def mix_normal_and_golden_samples(
    sample_json, golden_json, golden_fraction, total_output_reviews
):
    random.shuffle(sample_json)
    random.shuffle(golden_json)

    reduced_json = sample_json[:total_output_reviews]

    # every `step_size`th review will be replaced by a golden review s.t. the golden fraction is met
    step_size = round(1 / golden_fraction)

    print(
        f"Replacing every {step_size}th review (~{golden_fraction * 100}%) with a golden review..."
    )
    if len(reduced_json) / step_size > len(golden_json):
        print(f"WARNING: Some golden labels will be used more than once.")

    golden_idx = 0
    for idx_to_replace in range(0, len(reduced_json), step_size):
        golden_review = golden_json[golden_idx]
        reduced_json[idx_to_replace] = golden_review
        print(f"Inserted golden review: {golden_review['product_title']}")
        golden_idx = (golden_idx + 1) % len(golden_json)

    return reduced_json


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def parse_args(context):
    context.create_argument_parser(
        description="Generate a manifest file for a labelling job from a sampled file and our golden file. File format can be either tsv or json."
    )
    context.add_argument(
        "--sample-file", "-s", required=True, help="Sample tsv file name"
    )
    context.add_argument(
        "--golden-file",
        "-g",
        required=True,
        help="Golden dataset file name",
    )
    context.add_argument(
        "--number-of-tasks",
        "-n",
        default=20,
        type=int,
        help="The number of tasks (MTurk HITs) in the output manifest",
    )
    context.add_argument(
        "--reviews-per-task",
        "-r",
        default=10,
        type=int,
        help="The number of reviews for each task",
    )
    context.add_argument(
        "--golden-fraction",
        "-f",
        default=0.1,
        type=float,
        help="Proportion of golden samples",
    )
    context.add_argument(
        "--pre-label",
        "-p",
        choices=["existing", "openai", "none"],
        default="none",
        dest="PRE_LABEL",
        help="Create a pre-label manifest",
    )

    return context.parse_args(), context.format_help()


def main():
    with ReproducableContext("manifests/job", append_timestamp=False) as context:
        args, help_text = parse_args(context)

        if args.sample_file.endswith(".tsv"):
            sample_json = load_from_tsv(args.sample_file)
        elif args.sample_file.endswith(".json"):
            sample_json = load_from_json(args.sample_file)
        else:
            raise ValueError(
                "Expected sample dataset file to be in TSV or JSON format\n", help_text
            )

        if args.golden_file.endswith(".tsv"):
            golden_json = load_from_tsv(args.golden_file)
        elif args.golden_file.endswith(".json"):
            golden_json = load_from_json(args.golden_file)
        else:
            raise ValueError(
                "Expected golden dataset file to be in TSV or JSON format\n", help_text
            )
        reviews_per_task = args.reviews_per_task
        num_output_reviews = int(args.number_of_tasks) * int(args.reviews_per_task)

        if num_output_reviews > len(sample_json):
            print(
                f"WARNING: Only {len(sample_json)} samples are available which is less than requested."
            )
            num_output_reviews = len(sample_json)

        manifest = []
        SOURCE_COLUMN = "review_body"
        LABEL_COLUMN = "label"

        final_json = mix_normal_and_golden_samples(
            sample_json, golden_json, args.golden_fraction, num_output_reviews
        )
        for reviews_batch in grouper(final_json, reviews_per_task):
            reviews_batch = list(reviews_batch)
            random.shuffle(reviews_batch)
            source = []
            metadata = []
            for review in reviews_batch:
                source.append(review[SOURCE_COLUMN])

                datapoint_metadata = {}
                for column in review.keys():
                    if column != SOURCE_COLUMN and column != LABEL_COLUMN:
                        datapoint_metadata[column] = review[column]
                if args.PRE_LABEL != "none":
                    if args.PRE_LABEL == "existing":
                        datapoint_metadata["customUsageOptions"] = review["label"][
                            "customUsageOptions"
                        ]
                        datapoint_metadata["annotations"] = review["label"][
                            "annotations"
                        ]
                    else:
                        datapoint_metadata[
                            "customUsageOptions"
                        ] = pre_label_format_manifest(review)
                metadata.append(datapoint_metadata)

            manifest.append({"source": json.dumps(source), "metadata": metadata})

        json_string_dataset = [json.dumps(row, ensure_ascii=False) for row in manifest]
        formatted_json_string = "\n".join(json_string_dataset)

        with open(
            context.output_file(f"{context.id}-manifest.jsonl"), "w", encoding="utf8"
        ) as file:
            file.write(formatted_json_string)


if __name__ == "__main__":
    main()
