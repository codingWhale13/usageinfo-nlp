import abc
from helpers.review_set import ReviewSet
from helpers.review import Review
from active_learning.metrics.base import AbstractActiveLearningMetric
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import math


class AbstractSampler(abc.ABC):
    def __init__(self, n: int):
        self.n = n

    def sample(
        self, review_set: ReviewSet, metric_scores: dict[str, float]
    ) -> tuple[ReviewSet, float]:
        raise NotImplementedError()


class GreedySampler(AbstractSampler):
    def sample(
        self, review_set: ReviewSet, metric_scores: dict[str, float]
    ) -> tuple[ReviewSet, float]:
        print("Greedy sampling form the metric scores")
        start_time = time.time()
        sorted_scores = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        best_review_ids = [x[0] for x in sorted_scores[: self.n]]
        expected_information_gain = sum([x[1] for x in sorted_scores[: self.n]])
        sample = review_set.filter(
            lambda review: review.review_id in best_review_ids, inplace=False
        )
        print(f"Sampled {len(sample)} reviews in {time.time() - start_time}s")
        return sample, expected_information_gain


from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch


class GreedyOptimalClusterSubsetSampler(AbstractSampler):
    def __init__(self, include_existing_training_dataset: bool=False, **kwargs):
        super().__init__(*kwargs)
        self.include_existing_training_dataset = include_existing_training_dataset
    
    def sample(
        self, review_set: ReviewSet, metric_scores: dict[str, float]
    ) -> tuple[ReviewSet, float]:
        def calc_similarity_matrix(
            review_set_pool: list[Review],
            cluster_start_indices: dict[int, int],
            cluster_end_indices: dict[int, int],
        ):
            sentences = [
                review.get_prompt(prompt_id="active_learning_embedding_v2")
                for review in review_set_pool
            ]
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                # device="cuda",
            )
            embeddings = model.encode(
                sentences,
                show_progress_bar=True,
                batch_size=256,
                convert_to_tensor=True,
            ).cpu()

            similarities = []
            for cluster in tqdm(
                sorted(cluster_start_indices.keys()),
                desc="Calculating cosine similarities for each cluster",
            ):
                cluster_embeddings = embeddings[
                    cluster_start_indices[cluster] : cluster_end_indices[cluster] + 1
                ]
                similarities.append(
                    cosine_similarity(
                        cluster_embeddings.unsqueeze(1),
                        cluster_embeddings.unsqueeze(0),
                        dim=-1,
                    )
                )
            return similarities

        optimal_batch = []
        sorted_review_set_pool = []
        uncertainty_scores = []
        batch_scores = []
        TOTAL_POOL_SIZE = len(review_set)

        for _, (review_id, review) in enumerate(review_set.items()):
            sorted_review_set_pool.append(review)
            review_metric_score = metric_scores[review_id]
            if math.isnan(review_metric_score):
                review_metric_score = 0.0
            uncertainty_scores.append(review_metric_score)

        sorted_review_set_pool_with_uncertainty = sorted(
            zip(sorted_review_set_pool, uncertainty_scores),
            key=lambda r: r[0].data["product_category"],
        )

        sorted_review_set_pool, uncertainty_scores = zip(
            *sorted_review_set_pool_with_uncertainty
        )

        cluster_index = 0
        current_product_category = sorted_review_set_pool[0].data["product_category"]
        cluster_indices = {}
        cluster_start_indices = {0: 0}
        cluster_end_indices = {}
        for i_review, review in enumerate(sorted_review_set_pool):
            if review.data["product_category"] != current_product_category:
                cluster_end_indices[cluster_index] = i_review - 1
                cluster_index += 1
                cluster_start_indices[cluster_index] = i_review
                current_product_category = review.data["product_category"]

            cluster_indices[i_review] = cluster_index
        cluster_end_indices[cluster_index] = TOTAL_POOL_SIZE - 1

        # print(cluster_start_indices, cluster_end_indices)

        dp_max_similarity = torch.zeros(
            self.n + 1, TOTAL_POOL_SIZE, dtype=torch.float32
        )
        per_cluster_similarity_matrix = calc_similarity_matrix(
            sorted_review_set_pool, cluster_start_indices, cluster_end_indices
        )
        uncertainty_scores = torch.tensor(uncertainty_scores, dtype=torch.float32)

        def virtual_similarity_row(review_index):
            cluster_index = cluster_indices[review_index]
            cluster_start_index = cluster_start_indices[cluster_index]
            cluser_end_index = cluster_end_indices[cluster_index]
            similarities = torch.zeros(TOTAL_POOL_SIZE, dtype=torch.float32)
            similarities[
                cluster_start_index : cluser_end_index + 1
            ] = per_cluster_similarity_matrix[cluster_index][
                review_index - cluster_start_index
            ]
            return similarities

        def score_batch(
            batch_review_indices: list[int],
            uncertainty_scores: list[float],
        ):
            batch_round = len(batch_review_indices) - 1
            new_batch_review = batch_review_indices[-1]
            max_similarities = torch.max(
                dp_max_similarity[batch_round], virtual_similarity_row(new_batch_review)
            )
            return torch.dot(uncertainty_scores, max_similarities)

        optimal_batch_score = 0
        for _ in tqdm(
            range(self.n),
        ):
            batch_scores = []
            for j in range(TOTAL_POOL_SIZE):
                # print("Try:", j, "Score: ",score_batch(
                #        optimal_batch + [j],
                #        uncertainty_scores,
                #    ))
                batch_scores.append(
                    score_batch(
                        optimal_batch + [j],
                        uncertainty_scores,
                    )
                )

            new_batch_review = np.argmax(np.array(batch_scores))

            optimal_batch_score = batch_scores[new_batch_review]
            # print(new_batch_review, optimal_batch_score, optimal_batch)
            # print(batch_scores[new_batch_review], new_batch_review, cluster_indices[new_batch_review])
            optimal_batch.append(new_batch_review)
            batch_round = len(optimal_batch)
            for i_pool in range(TOTAL_POOL_SIZE):
                cluster_index = cluster_indices[i_pool]
                cluster_start_index = cluster_start_indices[cluster_index]
                dp_max_similarity[batch_round][i_pool] = max(
                    dp_max_similarity[batch_round - 1][i_pool],
                    0.0
                    if cluster_index != cluster_indices[new_batch_review]
                    else per_cluster_similarity_matrix[cluster_index][
                        i_pool - cluster_start_index
                    ][new_batch_review - cluster_start_index],
                )

        optimal_batch_review_ids = [
            sorted_review_set_pool[i].review_id for i in optimal_batch
        ]

        sample = review_set.filter(
            lambda review: review.review_id in optimal_batch_review_ids, inplace=False
        )
        return sample, optimal_batch_score

"""
class GreedyOptimalSubsetSampler(AbstractSampler):
    def sample(
        self, review_set: ReviewSet, metric_scores: dict[str, float]
    ) -> tuple[ReviewSet, float]:
        def calc_similarity_matrix(review_set_pool: list[Review]):
            sentences = [review.data["review_body"] for review in review_set_pool]
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cuda",
            )
            embeddings = model.encode(
                sentences,
                show_progress_bar=True,
                batch_size=256,
                convert_to_tensor=True,
            ).cpu()

            print("Calculating cosine similarity")
            return cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
            )

        optimal_batch = []
        review_set_pool = []
        uncertainty_scores = []
        batch_scores = []

        for review_id, review in review_set.items():
            review_set_pool.append(review)
            uncertainty_scores.append(metric_scores[review_id])

        dp_max_similarity = torch.zeros(
            self.n + 1, len(review_set_pool), dtype=torch.float32
        )
        similarity_matrix = calc_similarity_matrix(review_set_pool)
        uncertainty_scores = torch.tensor(uncertainty_scores, dtype=torch.float32)

        def score_batch(
            batch_review_indices: list[int],
            similarity_matrix,
            uncertainty_scores: list[float],
        ):
            batch_round = len(batch_review_indices) - 1
            new_batch_review = batch_review_indices[-1]
            max_similarities = torch.max(
                dp_max_similarity[batch_round], similarity_matrix[new_batch_review]
            )
            return torch.dot(uncertainty_scores, max_similarities)

        optimal_batch_score = 0
        for _ in tqdm(range(self.n)):
            batch_scores = []

            for j in range(len(review_set_pool)):
                batch_scores.append(
                    score_batch(
                        optimal_batch + [j],
                        similarity_matrix,
                        uncertainty_scores,
                    )
                )

            new_batch_review = np.argmax(np.array(batch_scores))
            optimal_batch_score = batch_scores[new_batch_review]
            optimal_batch.append(new_batch_review)
            batch_round = len(optimal_batch)
            for i_pool in range(len(review_set_pool)):
                dp_max_similarity[batch_round][i_pool] = max(
                    dp_max_similarity[batch_round - 1][i_pool],
                    similarity_matrix[i_pool][new_batch_review],
                )

        optimal_batch_review_ids = [review_set_pool[i].review_id for i in optimal_batch]

        sample = review_set.filter(
            lambda review: review.review_id in optimal_batch_review_ids, inplace=False
        )
        return sample, optimal_batch_score
"""