from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.module import ActiveLearningDataModule

from active_learning.metrics.base import AbstractActiveLearningMetric
from helpers.review_set import ReviewSet
from training.model import ReviewModel


class RandomAquisitionFunction(AbstractActiveLearningMetric):
    def __init__(self, n: int = None) -> None:
        self.n = n

    def compute(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ) -> dict[str, float]:
        scores = {}

        review_sample = review_set
        if self.n is not None:
            review_sample, _ = review_set.split(min(self.n / len(review_set), 1.0))

        for review_id in review_sample.reviews.keys():
            scores[review_id] = 1.0

        return scores

    def metric_name(self) -> str:
        return "random"
