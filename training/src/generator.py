import torch
import torch.nn.functional as F
from typing import List
from typing import Union

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
        checkpoint = torch.load(utils.get_model_path(self.model_artifact), map_location=torch.device('cpu'))
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
        predictions = self.tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]
        logprobs = self.get_logprobs(outputs)
        print(logprobs)
        return zip(review_ids, model_inputs, predictions, logprobs)
    
    def get_logprobs(self, outputs):
        # outputs["sequences"] are currently token ids, we need to decode every token. We slice off the first token because it is a pad token
        # (Könnte bomben if we use something else than T5 later)
        predicted_tokens = [[self.tokenizer.decode(token) for token in review[1:]] for review in outputs["sequences"]]

        # Use softmax to map to actual scores
        probs = [F.softmax(scores, dim=1) for scores in outputs["scores"]]

        # token_probs has review_number many entries, for every review we go through the relevant tokens and log their probability
        token_probs = [[probs[token_number][review_number, token].item() for token_number, token in enumerate(review[1:])] for review_number, review in enumerate(outputs["sequences"])]
        
        # outputs["sequences"] contains a lot of pad tokens, depending on the longest prediction in the batch (all predictions have equal length). We slice
        # predicted_tokens and token_probs as soon as we see a pad token to only log relevant tokens
        for i in range(len(predicted_tokens)):
            for j in range(len(predicted_tokens[i])):
                if predicted_tokens[i][j] == "<pad>":
                    predicted_tokens[i] = predicted_tokens[i][:j]
                    token_probs[i] = token_probs[i][:j]
                    break

        batch_probs = [{"predicted_tokens": predicted_tokens[i], "token_probs": token_probs[i]} for i in range(len(predicted_tokens))]

        return batch_probs


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


        if verbose:
            print(f"Generating label {label_id}...")

        for batch in dataloader:
            usage_options_batch = self.generate_usage_options(batch)
            for review_id, model_input, usage_options, logprobs in usage_options_batch:

                label_metadata = {
                "generator": {
                "artifact_name": self.model_artifact["name"],
                "checkpoint": self.model_artifact["checkpoint"] or "last",
                "generation_config": self.generation_config,
                    }
                }

                if label_id is not None:
                    label_metadata.update({"logprobs": logprobs})
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
