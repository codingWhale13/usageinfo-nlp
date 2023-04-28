import torch
import torch.nn.functional as F
from typing import List
from typing import Union
import numpy as np

from training import utils
from helpers.review_set import ReviewSet


class Generator:
    def __init__(
        self,
        artifact_name,
        checkpoint: int,
        generation_config: dict,
    ) -> None:
        self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}
        checkpoint = torch.load(
            utils.get_model_path(self.model_artifact),
            map_location=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),  # alternatively, map_location=torch.device("cpu") for CPU
        )
        model_config = utils.get_model_config_from_checkpoint(
            checkpoint["model"], checkpoint
        )

        self.model, self.tokenizer, self.max_length = model_config
        self.choices_per_step = generation_config["choices_per_step"]
        del generation_config["choices_per_step"]
        self.generation_config = generation_config
        self.flat = utils.get_config_from_artifact(artifact_name)["flat"]

    def format_usage_options(self, text_completion: str) -> List[str]:
        return [
            usage_option.strip()
            for usage_option in text_completion.split(", ")
            if usage_option.strip()
        ]

    def generate_usage_options(self, batch) -> None:
        # batch is Iterable containing [List of model inputs, List of labels, List of review_ids]
        review_ids = list(batch["review_id"])
        model_inputs = self.tokenizer.batch_decode(
            batch["input"]["input_ids"], skip_special_tokens=True
        )

        predicted_usage_options, predictions = self.generate(batch)

        if self.flat:
            for review_number, _ in enumerate(review_ids):
                covered_prod = predictions[review_number]["0.0"]["prob"]
                tested = 1

                while covered_prod < 0.5 and tested < 5:
                    new_decoder_input_ids = self.tokenizer(
                        predictions[review_number][f"0.{tested}"]["token"],
                        return_tensors="pt",
                    )["input_ids"]
                    usage_options, pred = self.generate(
                        batch, new_decoder_input_ids, offset=1
                    )
                    predicted_usage_options[review_number].extend(
                        usage_options[review_number]
                    )
                    predictions[review_number][f"0.{tested}"]["predictions"] = pred
                    covered_prod += predictions[review_number][f"0.{tested}"]["prob"]
                    tested += 1

        return review_ids, model_inputs, predicted_usage_options, predictions

    def generate(self, batch, decoder_input_ids=None, offset=0):
        return self.get_predictions_and_token_probs(
            self.model.generate(
                **batch["input"],
                **self.generation_config,
                decoder_input_ids=decoder_input_ids,
            ),
            offset,
        )

    def get_predictions_and_token_probs(self, outputs, offset):
        predicted_usage_options = self.tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=True
        )
        predicted_usage_options = [
            self.format_usage_options(usage_options)
            for usage_options in predicted_usage_options
        ]

        # Use softmax to map to actual scores
        probs = [F.softmax(scores, dim=1) for scores in outputs["scores"]]

        best_tokens = [
            [
                torch.argsort(
                    probs[token_number][review_number, :], dim=-1, descending=True
                )[: self.choices_per_step]
                for token_number, _ in enumerate(review[1:])
            ]
            for review_number, review in enumerate(outputs["sequences"])
        ]

        best_probs = [
            [
                [
                    probs[token_number][
                        review_number, best_tokens[review_number][token_number][choice]
                    ]
                    for choice in range(self.choices_per_step)
                ]
                for token_number, _ in enumerate(review[1:])
            ]
            for review_number, review in enumerate(outputs["sequences"])
        ]

        predictions = []
        for review_number, review in enumerate(outputs["sequences"]):
            prediction = {}
            for token_number in reversed(range(0, len(review) - 1)):
                pred = {}
                tokens = best_tokens[review_number][token_number]
                for choice, token in enumerate(tokens):
                    pred[f"{token_number + offset}.{choice}"] = {
                        "token": self.tokenizer.decode(token),
                        "prob": best_probs[review_number][token_number][choice].item(),
                    }

                if prediction != {}:
                    pred[f"{token_number+offset}.0"]["predictions"] = prediction
                prediction = pred
            predictions.append(prediction)

        return predicted_usage_options, predictions

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        if not label_id and not verbose:
            raise ValueError(
                "Specify either a label_id to save the labels or set verbose to True. (Or both)"
            )

        dataloader = reviews.get_dataloader(
            batch_size=1,
            num_workers=2,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            for_training=False,
            shuffle=True,
        )

        if verbose:
            print(f"Generating label {label_id}...")

        for batch in dataloader:
            (
                review_id,
                model_input,
                usage_options,
                predictions,
            ) = self.generate_usage_options(batch)
            label_metadata = {
                "generator": {
                    "artifact_name": self.model_artifact["name"],
                    "checkpoint": self.model_artifact["checkpoint"] or "last",
                    "generation_config": self.generation_config,
                },
                "predictions": predictions,
            }

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
                print(f"Label Metadata: {label_metadata}", end="\n\n")
