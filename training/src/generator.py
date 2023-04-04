import torch
from typing import List

from training.src import utils
from helpers.review_set import ReviewSet


class Generator:
    def __init__(
        self,
        artifact_name,
        checkpoint: int,
        generation_config: dict,
    ) -> None:
        self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}
        checkpoint = torch.load(utils.get_model_path(self.model_artifact))
        model_config = utils.get_model_config_from_checkpoint(
            checkpoint["model"], checkpoint
        )

        self.model, self.tokenizer, self.max_length = model_config
        self.generation_config = generation_config

    def format_usage_options(self, text_completion: str) -> List[str]:
        return [
            usage_option.strip()
            for usage_option in text_completion.split(", ")
            if usage_option.strip()
        ]

    def generate_usage_options(self, batch) -> None:
        # batch is Iterable containing [List of model inputs, List of labels, List of review_ids]
        review_ids = list(batch[2])
        model_inputs = self.tokenizer.batch_decode(
            batch[0]["input_ids"], skip_special_tokens=True
        )

        outputs = self.model.generate(**batch[0], **self.generation_config)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]

        return zip(review_ids, model_inputs, predictions)

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        if not label_id and not verbose:
            raise ValueError(
                "Specify either a label_id to save the labels or set verbose to True. (Or both)"
            )

        dataloader = reviews.get_dataloader(
            batch_size=8,
            num_workers=2,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            for_training=False,
        )
        label_metadata = {
            "generator": {
                "artifact_name": self.model_artifact["name"],
                "checkpoint": self.model_artifact["checkpoint"] or "last",
                "generation_config": self.generation_config,
            }
        }

        if verbose:
            print(f"Generating label {label_id}...")
            print(f"Label Metadata: {label_metadata}", end="\n\n")

        for batch in dataloader:
            usage_options_batch = self.generate_usage_options(batch)
            for review_id, model_input, usage_options in usage_options_batch:
                if label_id is not None:
                    reviews[review_id].add_label(
                        label_id=label_id,
                        usage_options=usage_options,
                        metadata=label_metadata,
                    )
                if verbose:
                    print(f"Review {review_id}")
                    print(f"Model input:\n{model_input}")
                    print(f"Usage options:\n\t{usage_options}", end="\n\n")
