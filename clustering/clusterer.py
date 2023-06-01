from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances_argmin_min

from typing import Dict
import numpy as np
import pandas as pd


class Clusterer:
    def __init__(
        self,
        review_set_df: pd.DataFrame,
        config: Dict,
        n_clusters: int,
    ):
        self.review_set_df = review_set_df
        self.config = config
        self.n_clusters = n_clusters

    def cluster(self):
        """
        This method performs clustering on the data using the specified clustering algorithm.
        """
        key = (
            "reduced_embedding"
            if "reduced_embedding" in self.review_set_df.columns
            else "embedding"
        )
        cluster_data = np.stack(self.review_set_df[key].to_numpy())

        if self.config["algorithm"] == "kmeans":
            return self.kmeans(cluster_data)
        elif self.config["algorithm"] == "agglomerative":
            return self.agglomerative(cluster_data)
        else:
            raise ValueError(
                f"Unknown clustering algorithm '{self.config['algorithm']}'"
            )

    def kmeans(self, cluster_data: np.ndarray):
        """
        This method performs clustering using k-means.

        Returns:
            labels (np.ndarray): Array of the cluster labels for each usage option.
            centroids (np.ndarray): The indices of the usage options that are the centroids of each cluster.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto").fit(
            cluster_data
        )

        centroids, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, cluster_data
        )
        # add centroids and labels to dataframe
        self.review_set_df["centroid"] = self.review_set_df.index.isin(centroids)
        self.review_set_df["label"] = kmeans.labels_.tolist()
        return self.review_set_df

    def agglomerative(self, cluster_data: np.ndarray):
        """
        This method performs clustering using agglomerative clustering.

        Returns:
            labels (np.ndarray): Array of the cluster labels for each usage option.
            centroids (np.ndarray): The indices of the usage options that are the best representative for each cluster.
        """
        agglomerative = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.config["metric"],
            linkage=self.config["linkage"],
        ).fit(cluster_data)

        clf = NearestCentroid(metric=self.config["metric"])
        clf.fit(cluster_data, agglomerative.labels_)
        centroids, _ = pairwise_distances_argmin_min(clf.centroids_, cluster_data)
        # add centroids and labels to dataframe
        self.review_set_df["centroid"] = self.review_set_df.index.isin(centroids)
        self.review_set_df["label"] = agglomerative.labels_.tolist()
        return self.review_set_df
