from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances_argmin_min

from typing import Dict
import numpy as np


class Clusterer:
    def __init__(
        self, embedded_usage_options: np.ndarray, config: Dict, n_clusters: int
    ):
        self.data = embedded_usage_options
        self.config = config
        self.n_clusters = n_clusters

    def cluster(self):
        """
        This method performs clustering on the data using the specified clustering algorithm.
        """
        if self.config["algorithm"] == "kmeans":
            return self.kmeans()
        elif self.config["algorithm"] == "agglomerative":
            return self.agglomerative()
        else:
            raise ValueError(
                f"Unknown clustering algorithm '{self.config['algorithm']}'"
            )

    def kmeans(self):
        """
        This method performs clustering using k-means.

        Returns:
            labels (np.ndarray): Array of the cluster labels for each usage option.
            centroids (np.ndarray): The indices of the usage options that are the centroids of each cluster.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto").fit(
            self.data
        )

        centroids, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, self.data)
        return kmeans.labels_, centroids

    def agglomerative(self):
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
        ).fit(self.data)

        clf = NearestCentroid(metric=self.config["metric"])
        clf.fit(self.data, agglomerative.labels_)
        centroids, _ = pairwise_distances_argmin_min(clf.centroids_, self.data)
        return agglomerative.labels_, centroids
