from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from helpers.review_set import ReviewSet
from training.model import ReviewModel


class ActiveLearningSupervisor:
    def __init__(self, model: ReviewModel, reviews: ReviewSet) -> None:
        self.training_dynamics = {"loss": []}
        self.model = model

        self.reviews = reviews
        self.individual_loss_function = CrossEntropyLoss(
            ignore_index=-100, reduction="mean"
        )

    def train_dataloader(self) -> DataLoader:
        if self.model.global_step() == 0:
            return self.__initial_train_dataloader()
        else:
            raise Exception(
                "Train dataloader not implemented for steps > 0. Check if reload_dataloaders_every_n_epochs is configured correctly"
            )

    def __initial_train_dataloader(self) -> DataLoader:
        return self.model.training_reviews().get_dataloader(
            **self.model.train_dataloader_args()
        )

    def reload_dataloaders_every_n_epochs(self) -> int:
        return 1000_000

    def process_step(self, batch_idx, batch, outputs) -> None:
        self.process_individual_losses(batch_idx, batch, outputs)

    def process_individual_losses(
        self, batch_idx, batch, outputs, mode="training"
    ) -> None:
        labels = batch["output"]["input_ids"]
        logits = outputs.logits.detach()

        logits = outputs.logits.detach()
        for i in range(logits.shape[0]):
            review_id = batch["review_id"][i]
            source_id = batch["source_id"][i]

            loss = self.individual_loss_function(logits[i, :, :], labels[i, :]).item()

            self.training_dynamics["loss"].append(
                {
                    "loss": loss,
                    "batch_idx": batch_idx,
                    "epoch": self.model.current_epoch(),
                    "mode": mode,
                    "step": self.model.global_step(),
                    "source_id": source_id,
                    "review_id": review_id,
                }
            )
