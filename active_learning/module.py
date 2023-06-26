from __future__ import annotations
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from typing import TYPE_CHECKING, Optional
from abc import ABCMeta

if TYPE_CHECKING:
    from helpers.review_set import ReviewSet
    from training.model import ReviewModel
from training.utils import get_model_dir_file_path
import pandas as pd

SAVE_EVERY_N_EPOCHS = 1


class AbstractActiveDataModule(metaclass=ABCMeta):
    def train_datalaoder(self) -> DataLoader:
        raise NotImplementedError

    def should_reset_train_dataloader(self) -> bool:
        raise NotImplementedError

    def process_step(self, batch_idx, batch, outputs, mode="training") -> None:
        raise NotImplementedError


class ActiveDataModule(AbstractActiveDataModule):
    def __init__(
        self, model: Optional[ReviewModel] = None, reviews: Optional[ReviewSet] = None
    ) -> None:
        self.training_dynamics = {"loss": []}
        self.model = model

        self.reviews = reviews
        self.individual_loss_function = CrossEntropyLoss(
            ignore_index=-100, reduction="mean"
        )

    def setup(self, model: ReviewModel, reviews: Optional[ReviewSet] = None) -> None:
        self.model = model
        self.reviews = reviews

    def train_dataloader(self) -> DataLoader:
        if self.model.global_step == 0:
            return self.__initial_train_dataloader()
        else:
            raise Exception(
                "Train dataloader not implemented for steps > 0. Check if reload_dataloaders_every_n_epochs is configured correctly"
            )

    def __initial_train_dataloader(self) -> DataLoader:
        dataloader, _ = self.model.training_reviews().get_dataloader(
            **self.model.train_dataloader_args()
        )
        return dataloader

    def process_step(self, batch_idx, batch, outputs, mode="training") -> None:
        self.process_individual_losses(batch_idx, batch, outputs, mode=mode)

    def should_reset_train_dataloader(self) -> bool:
        return False

    def dataframe(self, metric="loss") -> pd.DataFrame:
        return pd.DataFrame.from_records(self.training_dynamics[metric])

    def __save_training_dynamics(self):
        save_path = get_model_dir_file_path(
            self.model.run_name(), "training_dynamics.csv"
        )

        self.dataframe("loss").to_csv(save_path)

    def on_train_epoch_end(self):
        if self.should_reset_train_dataloader():
            self.model.trainer.reset_train_dataloader()
        if (self.model.current_epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            self.__save_training_dynamics()

    def process_individual_losses(self, batch_idx, batch, outputs, mode) -> None:
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


"""
class ActiveLearningLossBasedSampler(ActiveLearningModule):
    def train_dataloader(self):
        reviews = [
            (
                review.review_id,
                review["star_rating"],
                len(
                    review.get_label_from_strategy(
                        self.model.training_review_strategy()
                    )["usageOptions"]
                )
                > 0,
            )
            for review in self.model.training_reviews()
        ]
        df = pd.DataFrame(
            reviews, columns=["review_id", "star_rating", "has_usage_options"]
        )

        return df
df.groupby(["star_rating", "has_usage_options"]).count().groupby(["star_rating"]).min()
"""
