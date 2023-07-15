from __future__ import annotations
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Optional
from abc import ABCMeta
from helpers.review_set import ReviewSet
import wandb
from active_learning.dataset_analysis import analyze_review
from statistics import mean, variance, median
from scipy.stats import skew

if TYPE_CHECKING:
    from helpers.review_set import ReviewSet
    from training.model import ReviewModel
    from active_learning.metrics.base import AbstractActiveLearningMetric
    from active_learning.sampler import AbstractSampler

from training.utils import get_model_dir_file_path
import pandas as pd

SAVE_EVERY_N_EPOCHS = 1


class AbstractActiveDataModule(metaclass=ABCMeta):
    iteration = 0
    acquired_training_reviews = ReviewSet.from_reviews()
    unlabelled_training_reviews = ReviewSet.from_reviews()

    def _acquire_training_reviews(self) -> ReviewSet:
        raise NotImplementedError

    def acquired_training_reviews_size(self) -> int:
        return len(self.acquired_training_reviews)

    def log_dataframe(self, name: str, df: pd.DataFrame):
        save_path = get_model_dir_file_path(self.model.run_name(), f"{name}.csv")
        df.to_csv(save_path)
        self.model.logger.experiment.log(
            {
                "active_learning_iteration": self.iteration,
                name: wandb.Table(dataframe=df),
            }
        )

    def analyze_acquired_training_review(self):
        stats = []
        metric_name = self.metric.metric_name()
        for review in self.acquired_training_reviews:
            review_stats = analyze_review(review, self.model.train_reviews_strategy)

            stats.append(
                review_stats
                | {metric_name: self.current_metric_scores[review_stats["review_id"]]}
            )
        df = pd.DataFrame.from_records(data=stats, index="review_id")
        self.log_dataframe(f"training_dataset_iteration_{self.iteration}", df)

    def acquire_training_reviews(self) -> ReviewSet:
        self.model.sustainability_tracker.start(
            "active_learning_acquisition_function", iteration=self.iteration
        )
        new_training_reviews = self._acquire_training_reviews()
        self.acquired_training_reviews = (
            self.acquired_training_reviews | new_training_reviews
        )
        self.unlabelled_training_reviews = (
            self.unlabelled_training_reviews - new_training_reviews
        )
        self.iteration += 1

        self.model.sustainability_tracker.stop(
            "active_learning_acquisition_function", iteration=self.iteration
        )
        self.analyze_acquired_training_review()
        return self.acquired_training_reviews

    def should_reset_train_dataloader(self) -> bool:
        raise NotImplementedError

    def process_step(self, batch_idx, batch, outputs, mode="training") -> None:
        raise NotImplementedError

    def setup(self, model: ReviewModel, reviews: ReviewSet) -> None:
        self.model = model
        self.reviews = reviews
        """
        (
            self.unlabelled_validation_reviews,
            self.unlabelled_training_reviews,
        ) = reviews.split(1000 / len(reviews))
        """
        self.unlabelled_training_reviews = reviews
        self.unlabelled_validation_reviews = ReviewSet.from_reviews()
        print(
            f"Using {len(self.unlabelled_validation_reviews)} unlabeleld validation reviews and {len(self.unlabelled_training_reviews)} unlabeled training reviews"
        )

    def on_train_epoch_end(self) -> None:
        if len(self.unlabelled_validation_reviews) > 0:
            print("Calculating entropy on unlabelled validation set")
            scores = self.metric.compute(self.model, self.unlabelled_validation_reviews)
            entropy_scores = [entropy for entropy in scores.values()]
            self.model.logger.experiment.log(
                {
                    "trainer/global_step": self.model.trainer.global_step,
                    "validation_mean_entropy": mean(entropy_scores),
                    "validation_variance_entropy": variance(entropy_scores),
                }
            )
        if self.should_reset_train_dataloader():
            self.model.trainer.reset_train_dataloader()


class ActiveDataModule(AbstractActiveDataModule):
    def __init__(
        self,
        model: Optional[ReviewModel] = None,
        reviews: Optional[ReviewSet] = None,
        **kwargs,
    ) -> None:
        self.training_dynamics = {"loss": []}
        self.model = model

        self.reviews = reviews
        self.individual_loss_function = CrossEntropyLoss(
            ignore_index=-100, reduction="mean"
        )

    def setup(self, model: ReviewModel, reviews: Optional[ReviewSet] = None) -> None:
        super().setup(model, reviews)

    def acquire_training_reviews(self) -> ReviewSet:
        if self.model.global_step == 0:
            return self.model.train_reviews
        else:
            raise Exception(
                "Train dataloader not implemented for steps > 0. Check if reload_dataloaders_every_n_epochs is configured correctly"
            )

    def process_step(self, batch_idx, batch, outputs, mode="training") -> None:
        self.__process_individual_losses(batch_idx, batch, outputs, mode=mode)

    def __should_reset_train_dataloader(self) -> bool:
        return False

    def dataframe(self, metric="loss") -> pd.DataFrame:
        return pd.DataFrame.from_records(self.training_dynamics[metric])

    def __save_training_dynamics(self):
        save_path = get_model_dir_file_path(
            self.model.run_name(), "training_dynamics.csv"
        )

        self.dataframe("loss").to_csv(save_path)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if (self.model.current_epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            self.__save_training_dynamics()

    def __process_individual_losses(self, batch_idx, batch, outputs, mode) -> None:
        labels = batch["output"]["input_ids"]
        logits = outputs.logits.detach()

        for i in range(logits.shape[0]):
            review_id = batch["review_id"][i]
            source_id = batch["source_id"][i]

            loss = self.individual_loss_function(logits[i, :, :], labels[i, :]).item()

            self.training_dynamics["loss"].append(
                {
                    "loss": loss,
                    "batch_idx": batch_idx,
                    "epoch": self.model.current_epoch,
                    "mode": mode,
                    "step": self.model.global_step,
                    "source_id": source_id,
                    "review_id": review_id,
                }
            )


class ActiveLearningDataModule(AbstractActiveDataModule):
    current_metric_scores = {}

    def __init__(
        self,
        model: Optional[ReviewModel] = None,
        reviews: Optional[ReviewSet] = None,
        initial_samples: int = 1000,
        metric: AbstractActiveLearningMetric = None,
        sampler: AbstractSampler = None,
        training_epochs_per_iteration: int = 15,
    ) -> None:
        super().__init__()
        self.initial_samples = initial_samples
        self.metric = metric
        self.sampler = sampler
        self.training_dynamics = {"loss": []}
        self.individual_loss_function = CrossEntropyLoss(
            ignore_index=-100, reduction="mean"
        )
        self.training_epochs_per_iteration = training_epochs_per_iteration

    def process_step(self, batch_idx, batch, outputs, mode="training") -> None:
        self.__process_individual_losses(batch_idx, batch, outputs, mode=mode)

    def should_reset_train_dataloader(self) -> bool:
        print("Current epoch:", self.model.current_epoch)

        return (
            self.model.current_epoch != 0
            and (self.model.current_epoch % self.training_epochs_per_iteration) == 0
        )

    def _acquire_training_reviews(self) -> ReviewSet:
        metric_scores = self.metric.compute(
            self.model, self.unlabelled_training_reviews
        )
        metric_name = self.metric.metric_name()
        log_data = [
            {metric_name: score, "review_id": review_id}
            for review_id, score in metric_scores.items()
        ]
        raw_scores = metric_scores.values()
        self.log_dataframe(f"metric_scores_iteration_{self.iteration}", pd.DataFrame.from_records(log_data))
        self.model.logger.experiment.log({
            "active_learning_iteration": self.iteration,
            f"mean_{metric_name}": mean(raw_scores),
            f"median_{metric_name}": median(raw_scores)
            f"variance_{metric_name}": variance(raw_scores),
        })
        for review_id, score in metric_scores.items():
            try:
                self.current_metric_scores[review_id].append(score)
            except KeyError:
                self.current_metric_scores[review_id] = [score]
        return self.sampler.sample(self.unlabelled_training_reviews, metric_scores)

    def __process_individual_losses(self, batch_idx, batch, outputs, mode) -> None:
        labels = batch["output"]["input_ids"]
        logits = outputs.logits.detach()

        for i in range(logits.shape[0]):
            review_id = batch["review_id"][i]
            source_id = batch["source_id"][i]

            loss = self.individual_loss_function(logits[i, :, :], labels[i, :]).item()

            self.training_dynamics["loss"].append(
                {
                    "loss": loss,
                    "batch_idx": batch_idx,
                    "epoch": self.model.current_epoch,
                    "mode": mode,
                    "step": self.model.global_step,
                    "source_id": source_id,
                    "review_id": review_id,
                }
            )


# a null-object subclass of ActiveDataModule, which is used for test runs. all function calls should be no-ops
class NullActiveDataModule(AbstractActiveDataModule):
    def __init__(self) -> None:
        pass

    def train_dataloader(self):
        pass

    def __should_reset_train_dataloader(self):
        pass

    def process_step(self, batch_idx, batch, outputs, mode="training"):
        pass

    def setup(self, model: ReviewModel, reviews: ReviewSet):
        pass

    def on_train_epoch_end(self):
        pass


"""
class ActiveLearningLossBasedSampler(ActiveLearningModule):
    def train_dataloader(self):
        reviews = [
            (
                review.review_id,
                review["star_rating"],
                len(
                    review.get_label_from_strategy(
                        self.model.train_review_strategy
                    )["usageOptions"]
                )
                > 0,
            )
            for review in self.model.train_reviews
        ]
        df = pd.DataFrame(
            reviews, columns=["review_id", "star_rating", "has_usage_options"]
        )

        return df
df.groupby(["star_rating", "has_usage_options"]).count().groupby(["star_rating"]).min()
"""
