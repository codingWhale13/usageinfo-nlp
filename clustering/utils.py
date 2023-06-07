import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import yaml


def get_config(name: str) -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), rf"{name}.yml"
    )
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def plot_scores(scores: dict, output_path: str):
    """
    This method plots the scores for each number of clusters.

    Args:
        scores (dict): Dictionary of scores for each number of clusters.
        output_path (str): Path to save the plot to.
    """
    # convert to pandas dataframe
    df = pd.DataFrame(scores.items(), columns=["n_clusters", "score"])
    df2 = pd.concat([df["n_clusters"], pd.json_normalize(df["score"])], axis=1).drop(
        columns=["worst_cluster", "best_cluster", "calinski_harabasz", "davies_bouldin"]
    )
    # plot
    df2.plot(
        x="n_clusters", y=["silhouette", "avg_sim_in_cluster", "avg_sim_to_centroid"]
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.savefig(output_path)


def plot_clusters2d(clustered_df, n_clusters, color="label", interactive=False):
    df = pd.DataFrame(
        clustered_df,
        columns=["reduced_embedding", "label", "product_category", "usage_option"],
    )
    df["x"] = df["reduced_embedding"].apply(lambda x: x[0])
    df["y"] = df["reduced_embedding"].apply(lambda x: x[1])

    if not os.path.exists("plots"):
        os.mkdir("plots")

    if interactive:
        fig = px.scatter(df, x="x", y="y", color=color, hover_data=["usage_option"])
        fig.write_html(f"plots/plot{n_clusters}-{color}.html")
    else:
        plt.clf()
        sns.scatterplot(x="x", y="y", hue=color, data=df)
        plt.savefig(f"plots/plot{n_clusters}-{color}.png")
