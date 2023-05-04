from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

import itertools
import numpy as np


class Scorer:
    def __init__(
        self,
        embedded_usage_options: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ):
        self.data = embedded_usage_options
        self.labels = labels
        self.centroids = centroids
        self.cluster_distances = self.cluster_distances()

    def score(self):
        """
        This method calculates the scores for the clustering.
        """
        return {
            "silhouette": self.silhouette(),
            "davies_bouldin": self.davies_bouldin(),
            "calinski_harabasz": self.calinski_harabasz(),
            "avg_sim_in_cluster": self.average_sim_in_cluster(),
            "avg_sim_to_centroid": self.average_sim_to_centroid(),
            "worst_cluster": self.worst_cluster_index(),
            "best_cluster": self.best_cluster_index(),
        }

    def silhouette(self):
        return metrics.silhouette_score(self.data, self.labels)

    def davies_bouldin(self):
        return metrics.davies_bouldin_score(self.data, self.labels)

    def calinski_harabasz(self):
        return metrics.calinski_harabasz_score(self.data, self.labels)

    def cluster_distances(self):
        cluster_distances = [[] for i in range(len(self.centroids))]
        for i in range(len(self.centroids)):
            for j in range(len(self.data)):
                if self.labels[j] == i:
                    distance = cosine_similarity(
                        [self.data[self.centroids[i]]], [self.data[j]]
                    )[0][0]
                    cluster_distances[i].append(distance)
        return cluster_distances

    def average_sim_in_cluster(self):
        return np.mean([np.mean(cluster) for cluster in self.cluster_distances])

    def average_sim_to_centroid(self):
        distances = list(itertools.chain(*self.cluster_distances))
        return np.mean(distances)

    def worst_cluster_index(self):
        avg_cluster_distances = [np.mean(cluster) for cluster in self.cluster_distances]
        worst_index = avg_cluster_distances.index(min(avg_cluster_distances))
        return (worst_index, avg_cluster_distances[worst_index])

    def best_cluster_index(self):
        avg_cluster_distances = [np.mean(cluster) for cluster in self.cluster_distances]
        best_index = avg_cluster_distances.index(max(avg_cluster_distances))
        return (best_index, avg_cluster_distances[best_index])
