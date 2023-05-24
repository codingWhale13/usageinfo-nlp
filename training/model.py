import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader
from copy import copy
from typing import Optional

import training.utils as utils
from helpers.review_set import ReviewSet
from helpers.label_selection import (
    DatasetSelectionStrategy,
    LabelSelectionStrategyInterface,
)
from active_learning.module import ActiveLearningModule
from transformers.modeling_outputs import Seq2SeqLMOutput


NUM_WORKERS = 4


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
        seed: int,
        active_learning_module: ActiveLearningModule,
        lr_scheduler_type: Optional[str],
        optimizer_args: dict,
        gradual_unfreezing_mode: Optional[str],
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
        self.lr_scheduler_type = lr_scheduler_type
        self.optimizer_args = optimizer_args
        self.seed = seed
        self.gradual_unfreezing_mode = gradual_unfreezing_mode
        self.active_learning_module = active_learning_module
        self.tokenization_args = {
            "tokenizer": tokenizer,
            "model_max_length": max_length,
            "for_training": True,
        }

        self._initialize_datasets()

        self.active_learning_module.setup(self, self.reviews)

        # Skip freezing if a fake test model is loaded
        if self.model is not None:
            self._freeze_model()

    def _freeze_model(self):
        def unfreeze(component, slice_: str) -> int:
            transformer_blocks = eval(f"list(component.block)[{slice_}]")
            for block in transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = True
            # Returns the number of unfrozen transformer blocks
            return len(transformer_blocks)

        for param in self.model.parameters():
            param.requires_grad = False

        if self.active_layers["lm_head"]:
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

        self.active_encoder_layers = unfreeze(
            self.model.encoder, self.active_layers["encoder"]
        )
        self.active_decoder_layers = unfreeze(
            self.model.decoder, self.active_layers["decoder"]
        )
        self.gradual_unfreeze(0)

    def run_name(self) -> str:
        if self.trainer is not None and self.trainer.logger is not None:
            return self.trainer.logger.experiment.name
        return "test-run"

    def training_reviews(self) -> ReviewSet:
        return self.train_reviews

    def training_review_strategy(self) -> LabelSelectionStrategyInterface:
        return self.train_review_strategy

    def gradual_unfreeze(self, epoch: int):
        def unfreeze(blocks, epoch: int):
            for i in range(1, len(blocks) + 1):
                if epoch >= i:
                    for param in blocks[len(blocks) - i].parameters():
                        param.requires_grad = True

        def unfreeze_helper(active_layers, epoch, speed, module):
            epoch = epoch // speed + active_layers
            unfreeze(module, epoch)

        if self.gradual_unfreezing_mode is not None:
            unfreezing_modes = self.gradual_unfreezing_mode.split(", ")
            for mode in unfreezing_modes:
                if "encoder" in mode:
                    unfreeze_helper(
                        self.active_encoder_layers,
                        epoch,
                        int(mode.split(" ")[1]),
                        self.model.encoder.block,
                    )

                if "decoder" in mode:
                    unfreeze_helper(
                        self.active_decoder_layers,
                        epoch,
                        int(mode.split(" ")[1]),
                        self.model.decoder.block,
                    )

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

    def _step(self, batch) -> Seq2SeqLMOutput:
        # self() calls self.forward(), but should be preferred (https://github.com/Lightning-AI/lightning/issues/1209)
        labels = batch["output"]["input_ids"]
        outputs = self(
            input_ids=batch["input"]["input_ids"],
            attention_mask=batch["input"]["attention_mask"],
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        self.active_learning_module.process_step(batch_idx, batch, outputs)
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        return outputs.loss

    def on_train_epoch_end(self):
        print("On train epoch end")
        self.active_learning_module.on_train_epoch_end()

    def training_epoch_end(self, outputs):
        print("Self: training_epoch_end")
        """Logs the average training loss over the epoch"""
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "epoch_train_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        if self.lr_scheduler_type != None:
            self.log(
                "epoch_end_lr",
                self.lr_scheduler.get_last_lr()[0],
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=self.hyperparameters["batch_size"],
            )
        self.gradual_unfreeze(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        outputs = self._step(batch)
        self.active_learning_module.process_step(
            batch_idx, batch, outputs, mode="validation"
        )
        self.log(
            "val_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        return outputs.loss

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
            batch_size=self.hyperparameters["batch_size"],
        )

    def test_step(self, batch, __):
        outputs = self._step(batch)
        self.log(
            "test_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_args)

        if self.lr_scheduler_type == "OneCycleLR":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hyperparameters["max_lr"],
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        if self.lr_scheduler_type == "AdaFactor":
            from transformers.optimization import AdafactorSchedule

            self.lr_scheduler = AdafactorSchedule(optimizer)
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        # This is the right way to return an optimizer, without scheduler, in lightning (https://github.com/Lightning-AI/lightning/issues/3795)
        return [optimizer]

    def train_dataloader_args(self):
        return {
            **self.tokenization_args,
            "selection_strategy": self.train_review_strategy,
            "include_augmentations": True,
            "batch_size": self.hyperparameters["batch_size"],
            "drop_last": True,  # Drops the last incomplete batch, if the dataset size is not divisible by the batch size.
            "shuffle": True,  # Shuffles the training data every epoch.
            "num_workers": NUM_WORKERS,
            "multiple_usage_options_strategy": self.multiple_usage_options_strategy,
            "seed": self.seed,  # only relevant if shuffle=True,
            "pin_memory": True,
        }

    def train_dataloader(self):
        return self.train_reviews.get_dataloader(**self.train_dataloader_args())

    def val_dataloader(self):
        return self.val_reviews.get_dataloader(
            **self.tokenization_args,
            selection_strategy=self.train_review_strategy,
            include_augmentations=True,
            batch_size=self.hyperparameters["batch_size"],
            num_workers=NUM_WORKERS,
            multiple_usage_options_strategy=self.multiple_usage_options_strategy,
        )

    def test_dataloader(self):
        return self.test_reviews.get_dataloader(
            **self.tokenization_args,
            selection_strategy=self.test_reviews_strategy,
            include_augmentations=False,
            batch_size=self.hyperparameters["batch_size"],
            num_workers=NUM_WORKERS,
            multiple_usage_options_strategy=self.multiple_usage_options_strategy,
        )

    def _initialize_datasets(self):
        dataset_name = self.data["dataset_name"]
        self.test_reviews_strategy = DatasetSelectionStrategy((dataset_name, "test"))
        self.train_review_strategy = DatasetSelectionStrategy((dataset_name, "train"))

        self.reviews = ReviewSet.from_files(utils.get_dataset_path(dataset_name))
        self.test_reviews = self.reviews.filter_with_label_strategy(
            self.test_reviews_strategy, inplace=False
        )

        train_reviews = self.reviews.filter_with_label_strategy(
            self.train_review_strategy, inplace=False
        )

        self.val_reviews, self.train_reviews = train_reviews.split(
            self.data["validation_split"], seed=self.seed
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model"] = self.model_name
        return checkpoint
