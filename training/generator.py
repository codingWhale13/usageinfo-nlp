import torch
from typing import List
from typing import Union

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
        review_ids = list(batch["review_id"])
        model_inputs = self.tokenizer.batch_decode(
            batch["input"]["input_ids"], skip_special_tokens=True
        )

        outputs = self.model.generate(**batch["input"], **self.generation_config)
        print(outputs)
        predictions = self.tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
        print(predictions)
        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]
        logprobs = self.get_logprobs(outputs)
        return zip(review_ids, model_inputs, predictions, logprobs)
    
    def get_logprobs(self, outputs):
        predicted_tokens = [[self.tokenizer.decode(token) for token in review[1:]] for review in outputs["sequences"]]
        token_probs = [[outputs["scores"][token_number][review_number, token] for token_number, token in enumerate(review[1:])] for review_number, review in enumerate(outputs["sequences"])]
        
        for i in range(len(predicted_tokens)):
            for j in range(len(predicted_tokens[i])):
                if predicted_tokens[i][j] == "<pad>":
                    predicted_tokens[i] = predicted_tokens[i][:j]
                    token_probs[i] = token_probs[i][:j]
                    break
        print(predicted_tokens)
        print(token_probs)


        





    




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
            for review_id, model_input, usage_options, logprobs in usage_options_batch:
                if label_id is not None:
                    reviews[review_id].add_label(
                        label_id=label_id,
                        usage_options=usage_options,
                        metadata=label_metadata.update({"logprobs": logprobs}),
                    )
                if verbose:
                    print(f"Review {review_id}")
                    print(f"Model input:\n{model_input}")
                    print(f"Usage options:\n\t{usage_options}", end="\n\n")
