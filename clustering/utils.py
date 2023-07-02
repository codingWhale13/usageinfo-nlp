def get_config(name: str) -> dict:
    import os
    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), rf"{name}.yml"
    )
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def plot_scores(scores: dict, output_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

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


def plot_clusters2d(clustered_df, arg_dict, color="label", interactive=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns
    import plotly.express as px

    if arg_dict["n_clusters"] is not None:
        key = f'nclusters-{arg_dict["n_clusters"]}'
    elif arg_dict["distance_threshold"] is not None:
        key = f'distance-{arg_dict["distance_threshold"]}'
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
        fig.write_html(f"plots/plot{key}-{color}.html")
    else:
        plt.clf()
        sns.scatterplot(x="x", y="y", hue=color, data=df)
        plt.savefig(f"plots/plot{key}-{color}.png")


def get_arg_dicts(clustering_config, reviewset_length):
    # remove n_clusters and distance_thresholds from clustering config
    single_params = clustering_config["clustering"].copy()

    if "n_clusters" in clustering_config["clustering"]:
        del single_params["n_clusters"]

        if "distance_thresholds" in single_params:
            raise ValueError(
                f"You stupid?! Don't you ever again specify both n_clusters and distance_thresholds in the clustering config!"
            )
        if reviewset_length < max(clustering_config["clustering"]["n_clusters"]):
            raise ValueError(
                f"Reviewset length ({reviewset_length}) is smaller than the maximum number of clusters ({max(clustering_config['clustering']['n_clusters'])})."
            )

        return [
            {"n_clusters": n_clusters, "distance_threshold": None, **single_params}
            for n_clusters in clustering_config["clustering"]["n_clusters"]
        ]
    elif "distance_thresholds" in clustering_config["clustering"]:
        del single_params["distance_thresholds"]
        return [
            {
                "distance_threshold": distance_threshold,
                "n_clusters": None,
                **single_params,
            }
            for distance_threshold in clustering_config["clustering"][
                "distance_thresholds"
            ]
        ]
    else:
        raise ValueError(
            "Clustering config must contain either 'n_clusters' or 'distance_thresholds'"
        )


def merge_duplicated_usage_options(clustered_df, review_set_df):
    import pandas as pd

    return pd.merge(
        review_set_df,
        clustered_df[
            ["usage_option", "label", "centroid", "reduced_embedding"]
        ],  # need reduced embedding in the future
        on="usage_option",
        how="left",
    )


def save_clustered_df(clustered_df, arg_dict):
    if arg_dict["n_clusters"] is not None:
        key = f'nclusters-{arg_dict["n_clusters"]}'
    elif arg_dict["distance_threshold"] is not None:
        key = f'distance-{arg_dict["distance_threshold"]}'
    clustered_df[
        [
            "review_id",
            "usage_option",
            "product_id",
            "product_category",
            "reduced_embedding",
            "centroid",
            "label",
        ]
    ].to_csv(
        f"/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/data_clustering/clustered_usage_options/clustered_df_{key}.csv"
    )
