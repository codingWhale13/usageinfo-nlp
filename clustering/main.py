from data_loader import DataLoader
from clusterer import Clusterer
from scorer import Scorer
import argparse
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Perform clustering on the given reviewsets."
    )
    parser.add_argument(
        "reviewset_files",
        type=str,
        nargs="+",
        help="All reviewset files to cluster usage options from",
    )
    parser.add_argument(
        "label_id",
        type=str,
        help="Which label (aka which usage otpions) to use for clustering",
    )
    parser.add_argument(
        "-c",
        "--clustering_config",
        type=str,
        default="clustering_config",
        help="Clustering config to use",
    )

    return parser.parse_args(), parser.format_help()


def main():
    args, _ = arg_parse()
    label_id = args.label_id
    file_paths = args.reviewset_files
    clustering_config = utils.get_config(args.clustering_config)

    review_set_df = DataLoader(file_paths, label_id, clustering_config["data"]).load()

    if len(review_set_df) < max(clustering_config["clustering"]["n_clusters"]):
        raise ValueError(
            f"Number of usage options ({len(review_set_df)}) is smaller than the maximum number of clusters ({max(clustering_config['clustering']['n_clusters'])})"
        )

    scores = {}

    for n_clusters in clustering_config["clustering"]["n_clusters"]:
        print(f"Clustering with {n_clusters} clusters...")
        clustered_df = Clusterer(
            review_set_df,
            clustering_config["clustering"],
            n_clusters,
        ).cluster()

        if clustering_config["data"]["n_components"] == 2:
            utils.plot_clusters2d(
                clustered_df, n_clusters, color="label", interactive=False
            )

        print(f"Scoring {n_clusters} clusters...")
        scores[n_clusters] = Scorer(clustered_df).score()

    utils.plot_clusters2d(
        clustered_df, n_clusters, color="product_category", interactive=True
    )
    utils.plot_scores(scores, "scores.png")


if __name__ == "__main__":
    main()
