# %%%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_file", "-s", required=True, help="Sample tsv file name")
parser.add_argument(
    "--golden_file",
    "-g",
    required=True,
    help="Golden dataset file name",
)
parser.add_argument(
    "--number_of_tasks",
    "-n",
    default=10,
    help="The number of tasks (MTurk HITs) in the output manifest",
)
parser.add_argument(
    "--reviews_per_task",
    "-r",
    default=5,
    type=int,
    help="The number of reviews for each task",
)
parser.add_argument(
    "--golden_fraction",
    "-f",
    default=0.05,
    type=float,
    help="Proportion of golden samples",
)

args = parser.parse_args()

# %%
import json
import pandas as pd


def load_from_json(file_name):
    json_raw = json.load(open(file_name))
    df_temp = pd.DataFrame(json_raw["reviews"])
    return json.loads(df_temp.to_json(orient="records"))


def load_from_tsv(file_name):
    return json.loads(pd.read_csv(file_name, sep="\t").to_json(orient="records"))


if args.sample_file.endswith(".tsv"):
    sample_json = load_from_tsv(args.sample_file)
elif args.sample_file.endswith(".json"):
    sample_json = load_from_json(args.sample_file)
else:
    raise ValueError("Expected sample dataset file to be in TSV or JSON format")

if args.golden_file.endswith(".tsv"):
    golden_json = load_from_tsv(args.golden_file)
elif args.golden_file.endswith(".json"):
    golden_json = load_from_json(args.golden_file)
else:
    raise ValueError("Expected golden dataset file to be in TSV or JSON format")

# %%
num_output_reviews = int(args.number_of_tasks) * int(args.reviews_per_task)
if num_output_reviews > len(sample_json):
    print(
        f"WARNING: Only {len(sample_json)} samples are available which is less than requested."
    )
    num_output_reviews = len(sample_json)

# %%%
import random
from itertools import zip_longest

random.seed(42)


def mix_normal_and_golden_samples(
    sample_json, golden_json, golden_fraction, total_output_reviews
):
    random.shuffle(sample_json)
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
        reduced_json[idx_to_replace] = golden_json[golden_idx]
        golden_idx = (golden_idx + 1) % len(golden_json)

    return reduced_json


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


manifest = []
SOURCE_COLUMN = "review_body"

reviews_per_task = args.reviews_per_task

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
            if column != SOURCE_COLUMN:
                datapoint_metadata[column] = review[column]
        metadata.append(datapoint_metadata)

    manifest.append({"source": json.dumps(source), "metadata": metadata})

# %%
import time

json_string_dataset = [json.dumps(row, ensure_ascii=False) for row in manifest]
formatted_json_string = "\n".join(json_string_dataset)

unix_timestamp = str(int(time.time()))
with open(
    f"{args.sample_file}-{unix_timestamp}.manifest.jsonl", "w", encoding="utf8"
) as file:
    file.write(formatted_json_string)
