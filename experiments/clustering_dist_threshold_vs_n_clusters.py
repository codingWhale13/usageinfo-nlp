from clustering import utils
from clustering.data_loader import DataLoader
from clustering.clusterer import Clusterer
from helpers.review_set import ReviewSet

# This is an experiment to check the hypothesis
# "If we run experiment A using distance_treshold and experiment B using n_clusters
# and the results have the same number of clusters, the clusters are exactly the same.

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
file_path = (
    "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/data_labeled/experiments/1k.json"
)
rs = ReviewSet.from_files(file_path)
review_set_df = DataLoader(rs, "Golden", clustering_config["data"]).load()
arg_dicts = utils.get_arg_dicts(clustering_config, len(review_set_df))

for arg_dict in arg_dicts:
    clustered_df = Clusterer(review_set_df, arg_dict).cluster()

# 2) find out number of clusters
n_clusters = len(clustered_df["label"].unique())
print("NCLUSTERS", n_clusters)
print(clustered_df.head())

# 3) cluster with n_clusters and assert equality
arg_dict["n_clusters"] = n_clusters
arg_dict["distance_threshold"] = None
clustered_df_with_n_clusters = Clusterer(review_set_df, arg_dict).cluster()

assert clustered_df.equals(clustered_df_with_n_clusters), "Experiment failed!"

print("Experiment successful!")
