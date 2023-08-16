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
    def __init__(
        self,
        cluster_results: bool = False,
        prob_generator_max_sequence_length: int = 20,
        prob_generator_batch_size: int = 512,
        prob_generator_max_iterations: int = 100,
        prob_generator_minimum_probability: float = 0.001,
        prob_generator_minimum_total_probability: float = 0.95,
        prob_generator_token_top_k: int = 5,
        prompt_id: str = "active_learning_v1",
    ) -> None:
        self.cluster_results = cluster_results
        self.prob_generator_max_sequence_length = prob_generator_max_sequence_length
        self.prob_generator_batch_size = prob_generator_batch_size
        self.prob_generator_max_iterations = prob_generator_max_iterations
        self.prob_generator_minimum_probability = prob_generator_minimum_probability
        self.prob_generator_minimum_total_probability = (
            prob_generator_minimum_total_probability
        )
        self.prompt_id = prompt_id

        self.prob_generator_token_top_k = prob_generator_token_top_k
        super().__init__()

    def _compute_generations(
        self,
        active_learning_module: ActiveLearningDataModule,
        review_model: ReviewModel,
        review_set: ReviewSet,
    ):
        generator = BatchProbabilisticGenerator(
            model=review_model.model,
            tokenizer=review_model.tokenizer,
            max_sequence_length=self.prob_generator_max_sequence_length,
            batch_size=self.prob_generator_batch_size,
            max_iterations=self.prob_generator_max_iterations,
            minimum_probability=self.prob_generator_minimum_probability,
            minimum_total_probability=self.prob_generator_minimum_total_probability,
            token_top_k=self.prob_generator_token_top_k,
            prompt_id=self.prompt_id,
        )

        results = generator.generate_usage_options_prob_based_batch(
            review_set, cluster_results=self.cluster_results
        )
        df_data = []
        for review_id, review_data in results.items():
            for data in review_data:
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
