import abc
from helpers.review_set import ReviewSet
from active_learning.metrics.base import AbstractActiveLearningMetric
import time


class AbstractSampler(abc.ABC):
    def __init__(self, n: int):
        self.n = n

    def sample(
        self, review_set: ReviewSet, metric_scores: dict[str, float]
    ) -> ReviewSet:
        raise NotImplementedError()


class GreedySampler(AbstractSampler):
    def sample(
        self, review_set: ReviewSet, metric_scores: dict[str, float]
    ) -> ReviewSet:
        print("Greedy sampling form the metric scores")
        start_time = time.time()
        sorted_scores = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        best_review_ids = [x[0] for x in sorted_scores[: self.n]]

        sample = review_set.filter(
            lambda review: review.review_id in best_review_ids, inplace=False
        )
        print(f"Sampled {len(sample)} reviews in {time.time() - start_time}s")
        return sample
