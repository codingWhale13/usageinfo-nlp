# %%
from helpers.review_set import ReviewSet

review_set = ReviewSet.from_files(
    "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/training_artifacts/models/05_16_11_55_splendid-forest-441/reviews.json"
)
review_set
# %%
from helpers.label_selection import DatasetSelectionStrategy

train_strategy = DatasetSelectionStrategy(("burning-wood-31", "train"))
labels = [
    (review_id, review.get_label_from_strategy(train_strategy))
    for review_id, review in review_set.items()
]
labels

# %%
data_points = []
for review_id, label in labels:
    if label is not None:
        if "loss" in label["metadata"]:
            for data_point in label["metadata"]["loss"]:
                data_point["review_id"] = review_id
                data_point["has_usage_options"] = len(label["usageOptions"]) > 0
                data_points.append(data_point)

len(data_points)
# %%
import pandas as pd

df = pd.DataFrame(data_points)
df

# %%
from math import e
import seaborn as sns
from sklearn.cluster import OPTICS, SpectralClustering


# %%
def plot_dataset_map(
    df,
    clustering_model,
    min_epoch: int = 2,
    max_epoch: int = 15,
    mode: str = "any",
    usage_options_mode: str = "any",
):
    if mode == "training":
        df = df[df["mode"] == "training"]
    elif mode == "validation":
        df = df[df["mode"] == "validation"]
    elif mode != "any":
        raise ValueError("Mode must be any, training or validation")
    if usage_options_mode == "only_usage_options":
        df = df[df["has_usage_options"]]
    elif usage_options_mode == "only_no_usage_options":
        df = df[~df["has_usage_options"]]
    elif usage_options_mode != "any":
        raise ValueError(
            "usage_options_mode must be any, only_usage_options or only_no_usage_options"
        )

    df = df[df["epoch"] >= min_epoch]
    df = df[df["epoch"] <= max_epoch]
    df["loss_p"] = df["loss"].apply(lambda x: e ** (-x))

    stats = df.groupby(["review_id", "source_id", "has_usage_options"])["loss_p"].agg(
        ["mean", "std"]
    )

    X = stats.to_numpy()

    clustering_model.fit(X)
    stats["cluster"] = clustering_model.labels_

    sns.set_theme(rc={"figure.dpi": 300})
    return (
        sns.scatterplot(
            data=stats,
            x="std",
            y="mean",
            hue="source_id",
            style="has_usage_options",
            s=5,
            palette="deep",
        ),
        stats.reset_index(),
    )


clustering_model = SpectralClustering(n_clusters=3)

_, stats = plot_dataset_map(
    df,
    clustering_model,
    mode="training",
    min_epoch=1,
    usage_options_mode="any",
)
