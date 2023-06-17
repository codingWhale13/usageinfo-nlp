import torch
import torch.nn.functional as f
from typing import List, Optional
from typing import Union

from training import utils
from helpers.review_set import ReviewSet
from training.utils import get_config
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_GENERATION_CONFIG = "diverse_beam_search"

GENERATION_CONFIGS = [
    file_name.replace(".yml", "")
    for file_name in os.listdir(
        os.path.dirname(os.path.realpath(__file__)) + "/generation_configs/"
    )
]
from queue import PriorityQueue
from training import utils
from helpers.review_set import ReviewSet
from math import exp

QUEUE_PROBABILITY_OFFSET = 1000_000_000


class Generator:
    def __init__(
        self,
        artifact_name,
        generation_config: str = DEFAULT_GENERATION_CONFIG,
        checkpoint: Optional[Union[int, str]] = None,
        output_probabilities: str = "best",  # can be "all", "best" or None
    ) -> None:
        global device

        self.config = utils.load_config_from_artifact_name(artifact_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}
        (
            self.model,
            self.tokenizer,
            self.max_length,
            self.model_name,
        ) = utils.initialize_model_tuple(self.model_artifact)

        self.model.to(device)
        self.model.eval()
        self.generation_config = get_config(
            f"{os.path.dirname(os.path.realpath(__file__))}/generation_configs/{generation_config}.yml"
        )
        self.output_probabilities = output_probabilities

    def format_usage_options(self, text_completion: str) -> List[str]:
        return [
            usage_option.strip()
            for usage_option in text_completion.split(", ")
            if usage_option.strip()
        ]

    def get_elements_until_zero(self, numbers):
        result = []
        for num in numbers:
            if num == 0:
                break
            result.append(num)
        return result

    def generate_usage_options(self, batch) -> None:
        # batch is dict with keys: "input", "output", "review_id", "source_id"
        review_ids = list(batch["review_id"])
        input_ids = batch["input"]["input_ids"].to(self.device)
        attention_mask = batch["input"]["attention_mask"].to(self.device)
        model_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                num_return_sequences=1,
                return_dict_in_generate=True,
                **self.generation_config,
            )

        predictions = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]

        return zip(review_ids, model_inputs, predictions)

    def _get_output_with_probs(
        self, batch, forced_decoder_ids=[], current_probability=1.0
    ):
        res = [None for _ in range(len(predictions))]

        if self.output_probabilities == "best":
            predicted_tokens = [
                [self.tokenizer.decode(token) for token in review[1:]]
                for review in outputs["sequences"]
            ]

    def _get_output_with_probs(self, batch, generation_canidate: tuple[float, dict]):
        current_probability, canidate_info = generation_canidate
        forced_decoder_ids = canidate_info["forced_decoder_ids"]
        sequence_length = canidate_info["sequence_length"]

        num_token_options = 5
        max_new_tokens = len(forced_decoder_ids) + 1

        input_ids = batch["input"]["input_ids"].to(self.device)
        attention_mask = batch["input"]["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )
        # print(output)

        token_probs = f.softmax(output["scores"][-1][0], dim=0).log()
        top_k_token_ids, top_k_token_probs = torch.topk(
            token_probs, k=num_token_options, dim=0
        )

        for token_id in top_k_token_ids:
            yield (
                current_probability + top_k_token_probs[token_id],
                {
                    "forced_decoder_ids": forced_decoder_ids
                    + [(max_new_tokens, int(token_id))],
                    "sequence_length": sequence_length + 1,
                },
            )

    def generate_usage_options_prob_based(self, batch) -> None:
        results = []

        total_probability = 0
        MAX_ITERATIONS = 200

        generation_queue = PriorityQueue()

        generation_queue.put((0, {"forced_decoder_ids": [], "token_length": 0}))

        i = 0
        while (
            not generation_queue.empty()
            and i < MAX_ITERATIONS
            and total_probability <= 0.95
        ):
            next_generations = self._get_output_with_probs(
                batch, generation_queue.get()
            )
            i += 1

            for generation in next_generations:
                if generation[1]["forced_decoder_ids"][-1][1] == 1:
                    total_probability += exp(-generation[0])
                    results.append(generation)
                else:
                    generation_queue.put(generation)

        decoded_results = self.tokenizer.batch_decode(
            [
                [token_id for _, token_id in result[1]["forced_decoder_ids"]]
                for result in results
            ]
        )
        decoded_results_with_probs = [
            (
                exp(
                    -float(result[0])
                ),  # casting current_probability to float because it is a tensor
                result[1]["sequence_length"],
                output,
            )
            for result, output in zip(results, decoded_results)
        ]  # is a list with tuples (probability, length, prediction)

        return decoded_results_with_probs

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        if not label_id and not verbose:
            raise ValueError(
                "Specify either a label_id to save the labels or set verbose to True. (Or both)"
            )

        # you need to enforce batch size of 1 to allow proper calculation of probabilities; will slow down generation a lot
        if self.output_probabilities == "all":
            batch_size = 1
        else:
            batch_size = 32

        dataloader = reviews.get_dataloader(
            batch_size=batch_size,
            num_workers=0,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            for_training=False,
            prompt_id=self.config["prompt_id"],
        )

        label_metadata = {
            "generator": {
                "artifact_name": self.model_artifact["name"],
                "checkpoint": self.model_artifact["checkpoint"]
                if self.model_artifact["checkpoint"] is not None
                else "last",
                "model_name": self.model_name,
                "generation_config": self.generation_config,
            }
        }

        if verbose:
            print(f"Generating label {label_id}...")
            print(f"Label Metadata (base): {label_metadata}", end="\n\n")

        for batch in dataloader:
            if self.output_probabilities == "all":
                usage_options_batch = list(
                    self.generate_usage_options_prob_based(batch)
                )
            else:
                usage_options_batch = self.generate_usage_options(batch)

            for (
                review_id,
                model_input,
                usage_options,
                probabilities,
            ) in usage_options_batch:
                if label_id is not None:
                    if (
                        self.output_probabilities == "best"
                        or self.output_probabilities == "all"
                    ) and probabilities is not None:  # None check might be redundant
                        label_metadata["probabilities"] = probabilities
                    else:
                        # remove probabilities from metadata
                        label_metadata.pop("probabilities", None)
                    reviews[review_id].add_label(
                        label_id=label_id,
                        usage_options=usage_options,
                        metadata=dict(label_metadata),
                    )
                if verbose:
                    print(f"Review: {review_id}")
                    print(f"Model input:\n{model_input}")
                    print(f"Usage options:\n\t{usage_options}", end="\n\n")
                    print(f"Probabilities:\n\t{probabilities}", end="\n\n")
