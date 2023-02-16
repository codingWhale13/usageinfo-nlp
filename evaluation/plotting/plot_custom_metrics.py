import argparse
import json
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_f1(
    labels: list[dict],
    title: Optional[str] = None,
    save_path: Union[Path, str] = Path().absolute(),
    base_name: Optional[str] = "",
):
    sns.set_style("darkgrid")
    plt.clf()

    df = pd.DataFrame(labels)
    df = pd.concat([df, df["scores"].apply(pd.Series)], axis=1)
    sns.violinplot(
        data=df,
        x="custom_f1_score",
        cut=0,
    )

    if title is not None:
        plt.title(title)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_path, base_name + "f1.png"))


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
        "--labels",
        "-l",
        required=True,
        help="JSON file containing labels in the format described in README",
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

    with open(args.labels) as file:
        labels = json.load(file)
    multi_plot = args.multi_plot

    # check validity of labels
    if not (isinstance(labels, list) and isinstance(labels[0], dict)):
        raise ValueError("labels must be a list of dictionaries")
    if any("references" not in label for label in labels):
        raise ValueError("missing 'references' in labels")
    if any("predictions" not in label for label in labels):
        raise ValueError("missing 'predictions' in labels")
    if multi_plot and any("origin" not in label for label in labels):
        raise ValueError("'origin' must be specified in labels for multi-plot")

    # set save path to current directory if not specified
    save_path = args.save_path if args.save_path is not None else Path().absolute()
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # file name without extension will be the prefix for plot file names
    base_name = Path(args.labels).stem

    # plot
    plot_f1(labels=labels, save_path=save_path, base_name=base_name)
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
