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
from math import exp, log
import time

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

    def _get_ouput_with_probs_for_batch(
        self, generation_canidates: list[tuple[float, dict]]
    ):
        input_ids = torch.tensor([x["input_ids"] for x in generation_canidates[1]])
        decoder_input_ids = torch.tensor(
            [[z[1] for z in x["forced_decoder_ids"]] for x in generation_canidates]
        )

        attention_mask = torch.tensor(
            [x["attention_mask"] for x in generation_canidates]
        )

        with torch.no_grad():
            ouputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )

        raise NotImplementedError()

    def _get_output_with_probs(
        self, batch, generation_canidate: tuple[float, dict]
    ) -> list[tuple[float, dict]]:
        current_probability, canidate_info = generation_canidate
        forced_decoder_ids = canidate_info["forced_decoder_ids"]
        sequence_token_length = canidate_info["sequence_token_length"]

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

        token_probs = -(f.softmax(output["scores"][-1][0], dim=0).log())
        _, top_k_token_ids = torch.topk(-token_probs, k=num_token_options, dim=0)
        for token_id in top_k_token_ids:
            yield (
                current_probability + token_probs[int(token_id)],
                {
                    "forced_decoder_ids": forced_decoder_ids
                    + [(max_new_tokens, int(token_id))],
                    "sequence_token_length": sequence_token_length + 1,
                },
            )

    def generate_usage_options_prob_based(self, batch) -> list[list[dict]]:
        results = []

        total_probability = 0
        MAX_ITERATIONS = 100
        MINIMUM_PROBAILITY = -log(0.02)
        generation_queue = PriorityQueue()

        generation_queue.put(
            (0, {"forced_decoder_ids": [], "sequence_token_length": 0})
        )

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
                if generation[1]["forced_decoder_ids"][-1][1] == 1 or (
                    len(results) > 0 and generation[0] > MINIMUM_PROBAILITY
                ):
                    total_probability += exp(-generation[0])
                    results.append(generation)
                else:
                    generation_queue.put(generation)

        decoded_results = self.tokenizer.batch_decode(
            [
                [token_id for _, token_id in result[1]["forced_decoder_ids"]]
                for result in results
            ],
            skip_special_tokens=True,
        )
        decoded_results_with_probs = [
            [
                {
                    "probability": exp(
                        -float(result[0])
                    ),  # casting current_probability to float because it is a tensor
                    "sequence_token_length": result[1]["sequence_token_length"],
                    "usageOptions": self.format_usage_options(output),
                }
                for result, output in zip(results, decoded_results)
            ]
        ]
        return decoded_results_with_probs

    def generate_usage_options(self, batch: dict[str, list]) -> None:
        # batch is dict with keys: "input", "output", "review_id", "source_id"
        input_ids = batch["input"]["input_ids"].to(self.device)
        attention_mask = batch["input"]["attention_mask"].to(self.device)

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

        res = [None for _ in range(len(predictions))]

        predicted_tokens = [
            [self.tokenizer.decode(token) for token in review[1:]]
            for review in outputs["sequences"]
        ]

        # Use softmax to map to actual scores
        probs = [f.softmax(scores, dim=1) for scores in outputs["scores"]]

        token_probs = [
            torch.FloatTensor(
                [
                    probs[token_number][
                        review_number * self.generation_config["num_beams"]
                    ][
                        token
                    ]  # here we choose review_number * num_beams because we only want the first beam which is the best one for each review
                    for token_number, token in enumerate(review[1:])
                ]
            )
            for review_number, review in enumerate(outputs["sequences"])
        ]

        # outputs["sequences"] contains a lot of pad tokens, depending on the longest prediction in the batch (all predictions have equal length). We slice
        # predicted_tokens and token_probs as soon as we see a pad token to only log relevant tokens
        for i in range(len(predicted_tokens)):
            for j in range(len(predicted_tokens[i])):
                if predicted_tokens[i][j] == "<pad>":
                    predicted_tokens[i] = predicted_tokens[i][:j]
                    token_probs[i] = token_probs[i][:j]
                    break

        res = zip(
            [float(torch.prod(token_probs[i])) for i in range(len(predicted_tokens))],
            [len(predicted_tokens[i]) for i in range(len(predicted_tokens))],
            predictions,
        )

        res = [
            [
                {
                    "probability": x[0],
                    "sequence_token_length": x[1],
                    "usageOptions": x[2],
                }
            ]
            for x in res
        ]

        return res

    def pretty_print_format(self, probabilities: list[dict]):
        output = [
            (x["probability"], x["sequence_token_length"], x["usageOptions"])
            for x in probabilities
        ]
        for x in output:
            print(x)

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

            for review_id, usage_options in zip(
                batch["review_id"], usage_options_batch
            ):
                label_metadata["probabilities"] = usage_options
                if label_id is not None:
                    reviews[review_id].add_label(
                        label_id=label_id,
                        usage_options=usage_options[0]["usageOptions"],
                        metadata=dict(label_metadata),
                    )
                if verbose:
                    print(f"Review: {review_id}")
                    self.pretty_print_format(usage_options)
