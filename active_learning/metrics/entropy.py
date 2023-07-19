from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.module import ActiveLearningDataModule

from active_learning.metrics.base import AbstractActiveLearningMetric
from helpers.review_set import ReviewSet
from training.probablistic_generator import BatchProbabilisticGenerator
from training.model import ReviewModel
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np


def aggregate_probabilities(generations: list[dict]):
    aggregated_results = {}

    for x in generations:
        if tuple(set(x["usageOptions"])) not in aggregated_results:
            aggregated_results[tuple(set(x["usageOptions"]))] = x["probability"]
        else:
            aggregated_results[tuple(set(x["usageOptions"]))] += x["probability"]
    aggregated_results = [
        {"usageOptions": list(key), "probability": value}
        for key, value in aggregated_results.items()
    ]
    return aggregated_results


def calculate_normalized_entropy(probs: list[float]) -> float:
    if len(probs) == 0:
        return 0.0
    probs = np.array(probs)
    normalized_probs = probs / np.sum(probs)
    return -np.sum(normalized_probs * np.log2(normalized_probs))


def calculate_lowest_probability_approximation_entropy(probs: list[float]) -> float:
    if len(probs) == 0:
        return 0.0
    probs = np.array(probs)
    min_prob = np.min(probs)
    remaining_prob = 1 - sum(probs)
    k = int(remaining_prob / min_prob)
    last_remaining_prob = remaining_prob - k * min_prob
    return (
        -np.sum(probs * np.log(probs))
        - k * min_prob * np.log(min_prob)
        - (
            last_remaining_prob * np.log(last_remaining_prob)
            if last_remaining_prob != 0
            else 0
        )
    )


def load_entropy_approximation(entropy_approximation_name: str):
    entropy_functions = {
        "normalized": calculate_normalized_entropy,
        "lowest_probability_approximation": calculate_lowest_probability_approximation_entropy,
    }
    if entropy_approximation_name in entropy_functions:
        return entropy_functions[entropy_approximation_name]
    else:
        raise ValueError(
            f"entropy_approximation_name must be: {'', ''.join(entropy_functions.keys())}"
        )


class EntropyActiveLearningMetric(AbstractActiveLearningMetric):
    def __init__(
        self,
        decode_and_aggregate_results: bool = True,
        entropy_approximation: str = "normalized",
        prob_generator_max_sequence_length: int = 20,
        prob_generator_batch_size: int = 512,
        prob_generator_max_iterations: int = 100,
        prob_generator_minimum_probability: float = 0.001,
        prob_generator_minimum_total_probability: float = 0.95,
    ) -> None:
        super().__init__(decode_and_aggregate_results)
        self.entropy_approximation_function = load_entropy_approximation(
            entropy_approximation
        )
        self.prob_generator_max_sequence_length = prob_generator_max_sequence_length
        self.prob_generator_batch_size = prob_generator_batch_size
        self.prob_generator_max_iterations = prob_generator_max_iterations
        self.prob_generator_minimum_probability = prob_generator_minimum_probability
        self.prob_generator_minimum_total_probability = (
            prob_generator_minimum_total_probability
        )

    def compute(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ) -> dict[str, float]:
        results = self._compute_generations(
            active_learning_module, review_model, review_set
        )
        scores = {}
        print("Calculating entropy")

        for review_id, review in tqdm(results.items()):
            probs = [
                x["probability"]
                for x in (
                    aggregate_probabilities(review)
                    if self.decode_and_aggregate_results
                    else review
                )
            ]
            scores[review_id] = self.entropy_approximation_function(probs)

        return scores

    def metric_name(self) -> str:
        return "entropy"


class RandomEntropySampleActiveLearningMetric(AbstractActiveLearningMetric):
    def __init__(
        self,
        n: int = 4000,
        decode_and_aggregate_results: bool = True,
        entropy_approximation: str = "normalized",
    ) -> None:
        super().__init__(decode_and_aggregate_results)
        self.n = n
        self.entropy_approximation_function = load_entropy_approximation(
            entropy_approximation
        )

    def compute(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ) -> dict[str, float]:
        review_sample, _ = review_set.split(min(self.n / len(review_set), 1.0))
        print(f"Sampled subset of {len(review_sample)} reviews")
        results = self._compute_generations(
            active_learning_module, review_model, review_sample
        )
        scores = {}
        print("Calculating entropy")

        for review_id, review in tqdm(results.items()):
            probs = [
                x["probability"]
                for x in (
                    aggregate_probabilities(review)
                    if self.decode_and_aggregate_results
                    else review
                )
            ]
            scores[review_id] = self.entropy_approximation_function(probs)

        return scores

    def metric_name(self) -> str:
        return "random_sample_entropy"
