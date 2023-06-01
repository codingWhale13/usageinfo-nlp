from helpers.review_set import ReviewSet
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from evaluation.scoring.evaluation_cache import EvaluationCache
from evaluation.scoring.core import get_embedding
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import pandas as pd


class DataLoader:
    """
    This class is responsible for loading data from a list of files and encoding it
    using a pre-trained sentence embedding model. It can also perform dimensionality
    reduction using various methods.

    Args:
        file_paths (List[str]): A list of file paths to load data from.
        label_id (str): The label ID for the usage option data.
        config (Dict): A dictionary containing configuration options for the
            data loader, including the model name and the dimensionality reduction method.

    Returns:
        embedded_usage_options (numpy.ndarray): An array containing the encoded and
            possibly reduced usage option data.
    """

    def __init__(self, file_paths: List[str], label_id: str, config: Dict):
        self.file_paths = file_paths
        self.label_id = label_id
        self.model_name = config["model_name"]
        self.dim_reduction = config.get(
            "dim_reduction", None
        )  # Optional parameter for dimensionality reduction
        self.n_components = config.get("n_components", 2)

    def load(self):
        review_set = ReviewSet.from_files(*self.file_paths)

        embedded_usage_options = []
        review_set_list = []
        for review in review_set:
            for usage_option in review.get_usage_options(self.label_id):
                embedded_usage_option = get_embedding(
                    usage_option=usage_option, comparator=self.model_name
                )
                embedded_usage_options.append(embedded_usage_option)
                review_set_df.append(
                    {
                        "review_id": review.review_id,
                        "usage_option": usage_option,
                        "product_category": review["product_category"],
                        "embedding": embedded_usage_option,
                    }
                )
        review_set_df = pd.DataFrame(review_set_list)

        EvaluationCache.get().save_to_disk()

        reduced_usage_options = None

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
                reduced_usage_options = reducer.reduce(np.array(embedded_usage_options))
                review_set_df["reduced_embedding"] = reduced_usage_options.tolist()

            else:
                raise ValueError(
                    f"Unknown dimensionality reduction method '{self.dim_reduction}'"
                )

        return review_set_df


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
