import argparse
import os
import sys
import json
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = os.path.dirname(os.path.realpath(__file__))
new_path_split = path.split(os.sep)[:-2]
sys.path.append(os.path.join(os.path.sep, *new_path_split))

from utils.extract_reviews import extract_reviews_with_usage_options_from_json
from worker_metrics import Metrics


def calculate_golden_dataset_scores(
    vendor_file, golden_file, use_predicted_usage_options=False
):
    golden_df = extract_reviews_with_usage_options_from_json(golden_file)
    vendor_df = extract_reviews_with_usage_options_from_json(
        vendor_file, use_predicted_usage_options=use_predicted_usage_options
    )
    workers_df = vendor_df.groupby("workerId")

    scores = {}
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
        "false_negative"
    ]

    if not use_predicted_usage_options:
        for worker_id in workers_df.groups:
            worker_df = workers_df.get_group(worker_id)
            scores[worker_id] = Metrics(worker_df, golden_df).calculate(metrics)

    scores["total"] = Metrics(vendor_df, golden_df).calculate(metrics)

    return scores


def plot_metrics(metrics, title: str, save_path: Optional[str] = None):
    df = pd.DataFrame(metrics).drop("total", axis=1)
    df = df.loc[["custom_recall", "custom_precision"]]
    df = df.transpose().reset_index().rename({"index": "vendor"}, axis="columns")
    df = df.explode(["custom_precision", "custom_recall"]).reset_index(drop=True)

    sns.set_style("darkgrid")
    plt.clf()
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=df,
        x="custom_precision",
        y="custom_recall",
        alpha=0.6,  # overlapping points will be darker
        s=200,  #  use large point size for better visibility
        hue="vendor",
    )
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        "-l",
        required=True,
        help="JSON file containing manually labelled reviews",
    )
    parser.add_argument(
        "--golden",
        "-g",
        required=True,
        help="JSON file containing golden review labels",
    )

    parser.add_argument(
        "--predicted",
        "-p",
        help="If we want to use predicted usage",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--out-path",
        "-o",
        help="if specified, scores and image are saved here",
    )
    args = parser.parse_args()

    res = calculate_golden_dataset_scores(
        args.labels, args.golden, use_predicted_usage_options=args.predicted
    )

    file_base_name = f"{os.path.split(args.labels)[-1].split('.json')[0]}_score"

    if args.out_path is not None:
        with open(os.path.join(args.out_path, f"{file_base_name}.json"), "w") as file:
            json.dump(res, file)

    print("METRICS:")
    for worker, metrics in res.items():
        print(f"\n{worker}:")
        for metric_name, score in metrics.items():
            print(f"\t{metric_name}: {score}")

    """
    plot_metrics(
        res,
        title=file_base_name,
        save_path=os.path.join(args.out_path, f"{file_base_name}.png"),
    )
    """
