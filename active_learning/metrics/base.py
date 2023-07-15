from __future__ import annotations
import abc
from helpers.review_set import ReviewSet
from training.model import ReviewModel
from training.probablistic_generator import BatchProbabilisticGenerator
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.module import ActiveLearningDataModule


class AbstractActiveLearningMetric(abc.ABC):
    def __init__(self, decode_and_aggregate_results=True) -> None:
        self.decode_and_aggregate_results = decode_and_aggregate_results
        super().__init__()

    def _compute_generations(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ):
        generator = BatchProbabilisticGenerator(
            model=review_model.model, tokenizer=review_model.tokenizer
        )

        results = generator.generate_usage_options_prob_based_batch(
            review_set, decode_results=self.decode_and_aggregate_results
        )
        df_data = []
        for review_id, data_points in results.items():
            for data in data_points:
                df_data.append({"review_id": review_id, **data})

        active_learning_module.log_dataframe(
            f"generations_epoch_{review_model.current_epoch}",
            pd.DataFrame.from_records(df_data),
        )
        return results

    def compute(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ) -> dict[str, float]:
        raise NotImplementedError()

    def metric_name(self) -> str:
        raise NotImplementedError()
