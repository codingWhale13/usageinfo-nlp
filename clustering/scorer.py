import itertools

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity


class Scorer:
    def __init__(self, review_set_df: pd.DataFrame, path_to_golden_labels: str):
        self.clustered_df = review_set_df
        self.data = np.stack(review_set_df["embedding"].to_numpy())
        self.labels = review_set_df["label"].to_numpy()
        self.centroids = review_set_df[
            review_set_df["centroid"] == True
        ].index.to_numpy()
        self.cluster_distances = self.get_cluster_distances()
        self.golden_labels = self.get_golden_labels(path_to_golden_labels)

    def get_golden_labels(self, path):
        print(self.clustered_df)
        golden_labels_df = pd.read_csv(path, sep=";")
        golden_labels_df = golden_labels_df[["usage_option1", "usage_option2", "votes"]]
        # drop all rows where votes is "s"
        golden_labels_df = golden_labels_df[golden_labels_df["votes"] != "s"]
        # check wether usage_option1 or usage_option2 have the same label in the clustered_df
        golden_labels_df["cluster_vote"] = golden_labels_df.apply(
            lambda row: self.get_cluster_vote(row), axis=1
        )
        return golden_labels_df

    def get_cluster_vote(self, row):
        usage_option1 = row["usage_option1"]
        usage_option2 = row["usage_option2"]
        label1 = self.clustered_df[self.clustered_df["usage_option"] == usage_option1][
            "label"
        ].values[0]
        label2 = self.clustered_df[self.clustered_df["usage_option"] == usage_option2][
            "label"
        ].values[0]
        print(label1, label2)
        if label1 == label2:
            return True
        else:
            return False

    def get_cluster_distances(self):
        cluster_distances = [[] for _ in range(len(self.centroids))]
        for data_point_id, data_point in enumerate(self.data):
            for centroid_id, centroid in enumerate(self.centroids):
                if self.labels[data_point_id] == self.labels[centroid]:
                    X, Y = [self.data[centroid]], [data_point]
                    distance = cosine_similarity(X, Y)[0][0]
                    cluster_distances[centroid_id].append(distance)
        return cluster_distances

    def score(self):
        """
        This method calculates the scores for the clustering.
        """
        return {
            "silhouette": self.silhouette(),
            "avg_sim_to_centroid": self.average_sim_to_centroid(),
            "worst_cluster": self.worst_cluster_index(),
            "best_cluster": self.best_cluster_index(),
            "adjusted_rand": self.adjusted_rand(),
        }

    def adjusted_rand(self):
        cluster_labels = self.golden_labels["cluster_vote"].to_numpy()
        golden_labels = self.golden_labels["votes"].to_numpy()
        return metrics.adjusted_rand_score(cluster_labels, golden_labels)

    def silhouette(self):
        return metrics.silhouette_score(self.data, self.labels)

    def davies_bouldin(self):
        return metrics.davies_bouldin_score(self.data, self.labels)

    def calinski_harabasz(self):
        return metrics.calinski_harabasz_score(self.data, self.labels)

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
