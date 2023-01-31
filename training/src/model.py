import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader
import torchmetrics

import dataset as ds


class ReviewModel(pl.LightningModule):
    def __init__(
        self,
        model,
        model_name: str,
        tokenizer,
        max_length: int,
        optimizer,
        hparameters: dict,
        data: dict,
    ):
        super(ReviewModel, self).__init__()
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.optimizer = optimizer
        self.data = data
        self.hparameters = hparameters
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        self.val_dataset, self.train_dataset = self._get_dataset()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # self() calls self.forward(), but should be preferred (https://github.com/Lightning-AI/lightning/issues/1209)
        outputs = self(
            input_ids=batch[0]["input_ids"],
            attention_mask=batch[0]["attention_mask"],
            labels=batch[1]["input_ids"],
        )

        # outputs is a SequenceClassifierOutput object, which has a loss attribute at the first place (https://huggingface.co/docs/transformers/main_classes/output)
        return outputs.loss

    def training_step(self, batch, __):
        loss = self._step(batch)
        self.train_loss(loss)
        self.log(
            "train_loss",
            self.train_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_epoch_end(self, __):
        """Logs the average training loss over the epoch"""
        self.log(
            "epoch_train_loss",
            self.train_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        torch.cuda.empty_cache()

    def validation_step(self, batch, __):
        loss = self._step(batch)
        self.val_loss(loss)
        self.log(
            "val_loss",
            self.val_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_epoch_end(self, __):
        """Logs the average validation loss over the epoch"""
        self.log(
            "epoch_val_loss",
            self.val_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, __):
        loss = self._step(batch)
        self.log(
            "test_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # Huggingface recommends this optimizer https://github.com/facebookresearch/fairseq/blob/775122950d145382146e9120308432a9faf9a9b8/fairseq/optim/adafactor.py
        return [
            self.optimizer(
                self.parameters(),
                lr=self.hparameters["learning_rate"],
                weight_decay=self.hparameters["weight_decay"],
            )
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparameters["batch_size"],
            drop_last=True,  # Drops the last incomplete batch, if the dataset size is not divisible by the batch size.
            shuffle=True,  # Shuffles the training data every epoch.
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparameters["batch_size"],
            num_workers=2,
        )

    def test_dataloader(self):
        test_dataset = ds.ReviewDataset(
            self.data["test_dataset"],
            self.tokenizer,
            max_length=self.max_length,
        )
        return DataLoader(
            test_dataset,
            batch_size=self.hparameters["batch_size"],
            num_workers=2,
        )

    def _get_dataset(self):
        dataset = ds.ReviewDataset(
            self.data["train_dataset"],
            self.tokenizer,
            max_length=self.max_length,
        )
        return dataset.split(self.data["validation_split"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model"] = self.model_name
        return checkpoint
