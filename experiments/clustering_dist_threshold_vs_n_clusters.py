from clustering import utils
from clustering.data_loader import DataLoader
from clustering.clusterer import Clusterer

# This is an experiment to check the hypothesis
# "If we run experiment A using distance_treshold and experiment B using n_clusters and the results have the same number of clusters,
# the clusters are exactly the same.

clustering_config = {
    "data": {
        "model_name": "all-mpnet-base-v2",
        "dim_reduction": "tsne",
        "n_components": 2,
    },
    "clustering": {
        "use_reduced_embeddings": False,
        "algorithm": "agglomerative",
        "metric": "cosine",
        "linkage": "average",
        "save_to_disk": False,
        "distance_thresholds": [0.3],
    },
}

# 1) cluster with distance_threshold
file_path = "/home/codingwhale/Documents/Studium/HPI/Materialien/BP/bsc2022-usageinfo/data/comparison_easy_names.json"
review_set_df = DataLoader([file_path], "Golden", clustering_config["data"]).load()
arg_dicts = utils.get_arg_dicts(clustering_config, len(review_set_df))

for arg_dict in arg_dicts:
    clustered_df = Clusterer(
        review_set_df,
        clustering_config["clustering"],
        **arg_dict,
    ).cluster()
    print(clustered_df.head())

# 2) find out number of clusters
n_clusters = len(clustered_df["label"].unique())
print("NCLUSTERS", n_clusters)

# 3) cluster with n_clusters and assert equality
clustered_df_with_n_clusters = Clusterer(
    review_set_df,
    clustering_config["clustering"],
    n_clusters=n_clusters,
    distance_threshold=None,
).cluster()

assert clustered_df.equals(clustered_df_with_n_clusters), "WRONG"
print("Experiment successful!")
