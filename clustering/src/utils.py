import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd


def get_config(name: str) -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), rf"../{name}.yml"
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
