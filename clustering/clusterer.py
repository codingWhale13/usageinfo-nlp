import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestCentroid


class Clusterer:
    def __init__(
        self,
        review_set_df: pd.DataFrame,
        config: dict,
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
            if (
                "reduced_embedding" in self.review_set_df.columns
                and self.config["use_reduced_embeddings"]
            )
            else "embedding"
        )
        cluster_data = np.stack(self.review_set_df[key].to_numpy())

        if self.config["algorithm"] == "kmeans":
            return self.kmeans(cluster_data)
        elif self.config["algorithm"] == "agglomerative":
            return self.agglomerative(cluster_data)
        elif self.config["algorithm"] == "community_detection":
            return self.community_detection(cluster_data)
        else:
            raise ValueError(
                f"Unknown clustering algorithm '{self.config['algorithm']}'"
            )

    def kmeans(self, cluster_data: np.ndarray):
        """
        This method performs clustering using k-means.

        Returns:
            review_set_df (pd.DataFrame): The review set dataframe with the cluster labels and centroids added.
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
            review_set_df (pd.DataFrame): The review set dataframe with the cluster labels and centroids added.
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

    def community_detection(self, cluster_data: np.ndarray):
        """
        This method performs clustering using fast clustering.

        Returns:
            review_set_df (pd.DataFrame): The review set dataframe with the cluster labels and centroids added.
        """
        cluster_data = torch.tensor(cluster_data)
        clusters = util.community_detection(
            cluster_data, min_community_size=1, threshold=0.75
        )
        # add emtpy column "label" to dataframe
        self.review_set_df["label"] = np.nan
        # enumerate clusters and add labels to dataframe
        for i, cluster in enumerate(clusters):
            for index in cluster:
                self.review_set_df.at[index, "label"] = i
        labels = self.review_set_df["label"].tolist()
        # fit nearest centroid classifier and add centroids to dataframe
        clf = NearestCentroid(metric=self.config["metric"])
        clf.fit(cluster_data, labels)
        centroids, _ = pairwise_distances_argmin_min(clf.centroids_, cluster_data)
        self.review_set_df["centroid"] = self.review_set_df.index.isin(centroids)

        return self.review_set_df
