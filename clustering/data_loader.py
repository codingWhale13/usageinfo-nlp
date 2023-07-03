from typing import Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

import helpers.label_selection as ls
from evaluation.scoring.core import get_embedding
from evaluation.scoring.evaluation_cache import EvaluationCache
from helpers.review_set import ReviewSet


class DataLoader:
    """
    This class is responsible for loading data from a list of files and encoding it
    using a pre-trained sentence embedding model. It can also perform dimensionality
    reduction using various methods.

    Args:
        file_paths (list[str]): A list of file paths to load data from.
        label_id (str): The label ID for the usage option data.
        config (dict): A dictionary containing configuration options for the
            data loader, including the model name and the dimensionality reduction method.

    Returns:
        embedded_usage_options (numpy.ndarray): An array containing the encoded and
            possibly reduced usage option data.
    """

    def __init__(
        self,
        review_set: ReviewSet,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        config: dict,
    ):
        self.review_set = review_set
        self.label_selection_strategy = label_selection_strategy
        self.model_name = config["model_name"]
        self.dim_reduction = config.get("dim_reduction", None)  # optional parameter
        self.n_components = config.get("n_components", 2)

    def load(self):
        review_set_list = []
        for review in self.review_set:
            label_id = review.get_label_id_from_strategy(self.label_selection_strategy)
            if label_id is not None:
                for usage_option in review.get_usage_options(label_id):
                    if usage_option is not None and usage_option != "":
                        embedded_usage_option = get_embedding(
                            usage_option=usage_option.lower(),
                            comparator=self.model_name,
                        )
                        review_set_list.append(
                            {
                                "review_id": review.review_id,
                                "usage_option": usage_option,
                                "product_id": review["product_id"],
                                "product_category": review["product_category"],
                                "embedding": embedded_usage_option,
                            }
                        )
        review_set_df = pd.DataFrame(review_set_list)
        df_to_cluster = review_set_df.drop_duplicates(
            subset=["usage_option"], keep="first"
        ).copy()

        EvaluationCache.get().save_to_disk()

        if self.dim_reduction is not None:
            reducer_map = {
                "tsne": TSNEReducer,
                "pca": PCAReducer,
                "isomap": ISOMapReducer
                # Add more dimensionality reduction methods here
            }

            if self.dim_reduction in reducer_map:
                reducer_class = reducer_map[self.dim_reduction]
                reducer = reducer_class(self.n_components)
                reduced_usage_options = reducer.reduce(
                    np.array(df_to_cluster["embedding"].tolist())
                )
                df_to_cluster["reduced_embedding"] = reduced_usage_options.tolist()
            else:
                raise ValueError(
                    f"Unknown dimensionality reduction method '{self.dim_reduction}'"
                )

        return review_set_df, df_to_cluster


class TSNEReducer:
    """
    This class is responsible for reducing the dimensionality of data using t-SNE.

    Values for the hyperparameters of t-SNE are currently the defaults from scikit-learn, most notably:
        n_components=2
        perplexity=30.0
        early_exaggeration=12.0
        n_iter=1000

    Could be made configurable in the future.
    """

    def __init__(self, n_components):
        self.tsne = TSNE(n_jobs=-1, n_components=n_components, random_state=42)

    def reduce(self, data):
        print("Started reducing dimensions using t-SNE...")
        return self.tsne.fit_transform(data)


class ISOMapReducer:
    def __init__(self, n_components):
        self.isomap = Isomap(n_components=n_components, n_jobs=-1, n_neighbors=50)

    def reduce(self, data):
        print("Started reducing dimensions using ISOMap...")
        return self.isomap.fit_transform(data)


class PCAReducer:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def reduce(self, data):
        print("Started reducing dimensions using PCA...")
        return self.pca.fit_transform(data)
