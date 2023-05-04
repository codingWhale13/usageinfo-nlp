from helpers.review_set import ReviewSet
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from evaluation.scoring.evaluation_cache import EvaluationCache
from evaluation.scoring.core import get_embedding

from sklearn.manifold import TSNE


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

    def load(self):
        review_set = ReviewSet.from_files(*self.file_paths)
        usage_options = review_set.get_usage_options(self.label_id)
        unique_usage_options = list(set(map(lambda x: x.lower(), usage_options)))

        print(
            f"Loaded {len(unique_usage_options)} usage options from {len(self.file_paths)} files and now start embedding with {self.model_name}."
        )

        # embedd with eval cache
        embedded_usage_options = [
            get_embedding(usage_option=usage_option, comparator=self.model_name)
            for usage_option in unique_usage_options
        ]
        EvaluationCache.get().save_to_disk()

        if self.dim_reduction is not None:
            if self.dim_reduction == "tsne":
                reducer = TSNEReducer()
                embedded_usage_options = reducer.reduce(embedded_usage_options)

            # Add more dimensionality reduction methods here
            else:
                raise ValueError(
                    f"Unknown dimensionality reduction method '{self.dim_reduction}'"
                )

        return unique_usage_options, embedded_usage_options


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

    def __init__(self):
        self.tsne = TSNE(n_jobs=-1)

    def reduce(self, data):
        print("Started reducing dimensions using t-SNE...")
        return self.tsne.fit_transform(data)
