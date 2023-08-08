import argparse

from clusterer import Clusterer
from data_loader import DataLoader
from scorer import Scorer
from DBCV import DBCV
from scipy.spatial.distance import euclidean
import helpers.label_selection as ls
import utils
import pandas as pd
from helpers.review_set import ReviewSet
from sklearn.cluster import HDBSCAN
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Perform clustering on the given reviewsets"
    )
    parser.add_argument(
        "reviewset_files",
        type=str,
        nargs="+",
        help="All reviewset files to cluster usage options from",
    )
    parser.add_argument(
        "label_ids",
        type=str,
        help="Which label(s) (aka which usage options) to use for clustering",
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
    label_ids = args.label_ids.split(", ")
    files = args.reviewset_files
    review_set = ReviewSet.from_files(*files)
    clustering_config = utils.get_config(args.clustering_config)
    review_set_df, df_to_cluster = DataLoader(
        review_set, ls.LabelIDSelectionStrategy(*label_ids), clustering_config["data"]
    ).load()

    cluster_data = np.stack(df_to_cluster["embedding"].to_numpy())

    min_cluster_sizes = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
    scores = {}

    for min_cluster_size in min_cluster_sizes:
        hdbscan = HDBSCAN(
            min_samples=min_cluster_size,
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            store_centers="medoid",
        ).fit(cluster_data)

        # medoids is a list of embeddings
        df_to_cluster[f"centroid-{min_cluster_size}"] = df_to_cluster["embedding"].isin(
            hdbscan.medoids_.tolist()
        )
        df_to_cluster[f"label-{min_cluster_size}"] = hdbscan.labels_.tolist()
        # calculate DBCV score
        dbcv = DBCV(cluster_data, hdbscan.labels_, dist_function=euclidean)
        print(f"DBCV score for min_cluster_size {min_cluster_size}: {dbcv}")
        scores[min_cluster_size] = dbcv

    # save scores and review_set_df as csv
    df_to_cluster.to_csv(f"NEW_hdbscan_review_set_df.csv")
    pd.DataFrame.from_dict(scores, orient="index").to_csv(f"NEW_hdbscan_scores.csv")


if __name__ == "__main__":
    main()
