import time
import os
import argparse
import json
import pandas as pd

from worker_scores.worker_metrics import Metrics
from openai_pre_annotion import pre_label_format_manifest


metrics = [
    "recall",
    "specificity",
    "f1",
    "precision",
    "miss_rate",
    "accuracy",
    "balanced_accuracy",
    "true_negative",
    "true_positive",
    "false_positive",
    "false_negative",
    "custom_recall",
    "custom_precision",
]


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Evaluate various prompts and their OpenAI performance.")
    arg_parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="json file path with the golden labelled reviews")
    arg_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="json file path with prompts")

    return arg_parser.parse_args(), arg_parser.format_help()


def main():
    args, help_text = parse_args()

    reviews_file_name = os.path.abspath(args.file)
    prompts_file_name = os.path.abspath(args.prompts)

    with open(reviews_file_name, 'r') as reviews_file, open(prompts_file_name, 'r+') as prompts_file:
        review_json = json.load(reviews_file)
        prompts_json = json.load(prompts_file)

        for prompt in prompts_json:
            if prompt['name'] not in review_json['reviews'][0]['label'].keys():
                print(f"No existing OpenAI labels for prompt {prompt['name']} found. Starting api querying...")
                for review in review_json['reviews']:
                    review['label'][prompt['name']] = pre_label_format_manifest(
                        review, prompt=prompt["prompt"])
            else:
                print(f"Existing OpenAI labels for prompt {prompt['name']} found. Skipping api querying...")

            openai_df = pd.DataFrame(review_json['reviews'])
            openai_df["usage_options"] = openai_df["label"].apply(
                lambda x: x[prompt['name']])

            golden_df = pd.DataFrame(review_json['reviews'])
            golden_df["usage_options"] = golden_df["label"].apply(lambda x: x["customUsageOptions"] + [
                " ".join(annotation["tokens"]) for annotation in x["annotations"]
            ])

            prompt['score'] = Metrics(
                openai_df, golden_df).calculate(metrics)


        output_file_name = os.path.dirname(
            reviews_file_name) + f"/prompt_labelled_" + os.path.basename(reviews_file_name)
        with open(output_file_name, 'w') as reviews_file:
            json.dump(review_json, reviews_file)

        prompts_file.truncate(0)
        prompts_file.seek(0)
        json.dump(prompts_json, prompts_file)


if __name__ == "__main__":
    main()
