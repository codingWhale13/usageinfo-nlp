import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader
from copy import copy

import utils
from helpers.review_set import ReviewSet
from helpers.label_selection import DatasetSelectionStrategy


class ReviewModel(pl.LightningModule):
    def __init__(
        self,
        model,
        active_layers: str,
        model_name: str,
        tokenizer,
        max_length: int,
        optimizer,
        hyperparameters: dict,
        data: dict,
        trainer: pl.Trainer,
        multiple_usage_options_strategy: str,
    ):
        super(ReviewModel, self).__init__()
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.optimizer = optimizer
        self.data = data
        self.hyperparameters = hyperparameters
        self.active_layers = active_layers
        self.trainer = trainer
        self.multiple_usage_options_strategy = multiple_usage_options_strategy

        self.tokenization_args = {
            "tokenizer": tokenizer,
            "model_max_length": max_length,
            "for_training": True,
        }

        self._initialize_datasets()

        self._freeze_model()

    def _freeze_model(self):
        def unfreeze(component, slice_: str):
            transformer_blocks = eval(f"list(component.block)[{slice_}]")
            for block in transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = False

        if self.active_layers["lm_head"]:
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

        unfreeze(self.model.encoder, self.active_layers["encoder"])
        unfreeze(self.model.decoder, self.active_layers["decoder"])

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
            input_ids=batch["input"]["input_ids"],
            attention_mask=batch["input"]["attention_mask"],
            labels=batch["output"]["input_ids"],
        )

        # outputs is a SequenceClassifierOutput object, which has a loss attribute at the first place (https://huggingface.co/docs/transformers/main_classes/output)
        return outputs.loss

    def training_step(self, batch, __):
        loss = self._step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_epoch_end(self, outputs):
        """Logs the average training loss over the epoch"""
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "epoch_train_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "epoch_end_lr",
            self.lr_scheduler.get_last_lr()[0],
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        torch.cuda.empty_cache()

    def validation_step(self, batch, __):
        loss = self._step(batch)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_epoch_end(self, outputs):
        """Logs the average validation loss over the epoch"""
        avg_loss = torch.stack(outputs).mean()
        self.log(
            "epoch_val_loss",
            avg_loss,
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
        optimizer = self.optimizer(
            self.parameters(),
            weight_decay=self.hyperparameters["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hyperparameters["max_lr"],
            total_steps=self.trainer.estimated_stepping_batches,
        )
        self.lr_scheduler = lr_scheduler
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        return self.train_reviews.get_dataloader(
            **self.tokenization_args,
            selection_strategy=self.train_review_strategy,
            batch_size=self.hyperparameters["batch_size"],
            drop_last=True,  # Drops the last incomplete batch, if the dataset size is not divisible by the batch size.
            shuffle=True,  # Shuffles the training data every epoch.
            num_workers=2,
            multiple_usage_options_strategy=self.multiple_usage_options_strategy,
            dataset_name=self.data["dataset_name"],
        )

    def val_dataloader(self):
        return self.val_reviews.get_dataloader(
            **self.tokenization_args,
            selection_strategy=self.train_review_strategy,
            batch_size=self.hyperparameters["batch_size"],
            num_workers=2,
            multiple_usage_options_strategy=self.multiple_usage_options_strategy,
            dataset_name=self.data["dataset_name"],
        )

    def test_dataloader(self):
        return self.test_reviews.get_dataloader(
            **self.tokenization_args,
            selection_strategy=self.test_reviews_strategy,
            batch_size=self.hyperparameters["batch_size"],
            num_workers=2,
            multiple_usage_options_strategy=self.multiple_usage_options_strategy,
            dataset_name=self.data["dataset_name"],
        )

    def _initialize_datasets(self):
        dataset_name = self.data["dataset_name"]
        self.test_reviews_strategy = DatasetSelectionStrategy((dataset_name, "test"))
        self.train_review_strategy = DatasetSelectionStrategy((dataset_name, "train"))

        reviews = ReviewSet.from_files(utils.get_dataset_path(dataset_name))

        self.test_reviews = reviews.filter_with_label_strategy(
            self.test_reviews_strategy, inplace=False
        )

        train_reviews = reviews.filter_with_label_strategy(
            self.train_review_strategy, inplace=False
        )
        self.train_reviews, self.val_reviews = train_reviews.split(
            self.data["validation_split"]
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model"] = self.model_name
        return checkpoint
