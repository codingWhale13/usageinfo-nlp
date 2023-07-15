from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.module import ActiveLearningDataModule

from active_learning.metrics.base import AbstractActiveLearningMetric
from helpers.review_set import ReviewSet
from training.model import ReviewModel
from tqdm import tqdm
from active_learning.metrics.entropy import aggregate_probabilities


class LeastConfidenceActiveLearningMetric(AbstractActiveLearningMetric):
    def compute(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ) -> dict[str, float]:
        results = self._compute_generations(review_model, review_set)
        scores = {}
        print("Calculating least_confidence")

        for review_id, review in tqdm(results.items()):
            probs = [x["probability"] for x in aggregate_probabilities(review)]
            scores[review_id] = 1 - max(probs)

        return scores

    def metric_name(self) -> str:
        return "least_confidence"
