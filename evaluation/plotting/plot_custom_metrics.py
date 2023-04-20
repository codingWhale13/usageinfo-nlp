#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_scores(data: dict, ref_id: str, pred_id: str, metric_name: str):
    scores = []
    for review in data["reviews"]:
        if "scores" in review["labels"][pred_id]["metadata"]:
            scores.append(
                review["labels"][pred_id]["metadata"]["scores"][ref_id][metric_name]
            )
    return scores


def plot_f1(
    scores: list[float],
    score_name="f1 score",
    title: Optional[str] = None,
    save_path: Union[Path, str] = Path().absolute(),
    base_name: Optional[str] = "",
):
    sns.set_style("darkgrid")
    plt.clf()

    sns.violinplot(
        data=scores,
        cut=0,
    )

    if title is not None:
        plt.title(title)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path, base_name + f"_{score_name}.png"))


def kde_plot_precision_vs_recall(
    labels: list[dict],
    save_path: Union[Path, str] = Path().absolute(),
    base_name: Optional[str] = "",
    title: Optional[str] = None,
    multi_plot=False,
):
    sns.set_style("darkgrid")
    plt.clf()

    df = pd.DataFrame(labels)
    df = pd.concat([df, df["scores"].apply(pd.Series)], axis=1)

    if multi_plot:
        g = sns.FacetGrid(df, col="origin", col_wrap=5)
        g.map(
            sns.kdeplot, "custom_recall", "custom_precision", fill=True, alpha=1, cut=0
        )
    else:
        sns.kdeplot(
            df, x="custom_recall", y="custom_precision", fill=True, alpha=1, cut=0
        )

    if title is not None:
        plt.title(title)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path, base_name + "precision_vs_recall_kde.png"))


def scatter_plot_precision_vs_recall(
    labels: list[dict],
    save_path: Union[Path, str] = Path().absolute(),
    base_name: Optional[str] = "",
    title: Optional[str] = None,
    multi_plot=False,
):
    sns.set_style("darkgrid")
    plt.clf()

    df = pd.DataFrame(labels)
    df = pd.concat([df, df["scores"].apply(pd.Series)], axis=1)

    if multi_plot:
        g = sns.FacetGrid(df, col="origin", col_wrap=5)
        g.map(sns.scatterplot, "custom_recall", "custom_precision", alpha=0.5, s=100)
    else:
        sns.scatterplot(df, x="custom_recall", y="custom_precision", alpha=0.5, s=100)

    if title is not None:
        plt.title(title)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path, base_name + "precision_vs_recall_scatter.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        "-j",
        required=True,
        help="JSON file in our format v1",
    )
    parser.add_argument(
        "--multi-plot",
        "-m",
        default=False,
        help="if set, plots will differentiate between origins of labels",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        help="folder to save plots (default: current directory)",
    )
    args = parser.parse_args()

    with open(args.json) as file:
        data = json.load(file)
    multi_plot = args.multi_plot

    ref_id = "golden_v2"
    pred_id = "leo_k"
    metric_name = "custom_f1_score_min"
    scores = extract_scores(data, ref_id, pred_id, metric_name)

    # set save path to current directory if not specified
    save_path = args.save_path if args.save_path is not None else Path().absolute()
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # file name without extension will be the prefix for plot file names
    base_name = Path(args.json).stem

    # plot
    plot_f1(scores, metric_name, save_path=save_path, base_name=base_name)

    exit()  # other plots not updated yet to our JSON format v1...

    kde_plot_precision_vs_recall(
        labels=labels,
        save_path=save_path,
        multi_plot=multi_plot,
        base_name=base_name,
    )
    scatter_plot_precision_vs_recall(
        labels=labels,
        save_path=save_path,
        multi_plot=multi_plot,
        base_name=base_name,
    )
