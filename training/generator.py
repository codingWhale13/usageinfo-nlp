import torch
import torch.nn.functional as f
from typing import List, Optional
from typing import Union

from training import utils
from helpers.review_set import ReviewSet
from training.utils import get_config
import os
from queue import PriorityQueue
from math import exp, log

DEFAULT_GENERATION_CONFIG = "diverse_beam_search"

GENERATION_CONFIGS = [
    file_name.replace(".yml", "")
    for file_name in os.listdir(
        os.path.dirname(os.path.realpath(__file__)) + "/generation_configs/"
    )
]

from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class GenerationCanidate:
    probability: float
    data: Any = field(compare=False)


class Generator:
    MAX_SEQUENCE_LENGTH = 20

    def __init__(
        self,
        artifact_name,
        generation_config: str = DEFAULT_GENERATION_CONFIG,
        checkpoint: Optional[Union[int, str]] = None,
        output_probabilities: str = "best",  # can be "all", "best" or None
    ) -> None:
        self.config = utils.load_config_from_artifact_name(artifact_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}
        (
            self.model,
            self.tokenizer,
            self.max_length,
            self.model_name,
        ) = utils.initialize_model_tuple(self.model_artifact)

        self.model.to(self.device)
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

    def __get_output_with_probs(
        self, batch, generation_canidate: GenerationCanidate
    ) -> list[tuple[float, dict]]:
        current_probability = generation_canidate.probability
        canidate_info = generation_canidate.data
        decoder_input_ids = canidate_info["forced_decoder_ids"].to(self.device)

        sequence_token_length = canidate_info["sequence_token_length"]
        decoder_attention_mask = torch.zeros(
            [1, self.MAX_SEQUENCE_LENGTH + 1], dtype=torch.int32
        ).to(self.device)
        decoder_attention_mask[0][:sequence_token_length] = 1
        num_token_options = 5

        input_ids = batch["input"]["input_ids"].to(self.device)
        attention_mask = batch["input"]["attention_mask"].to(self.device)

        encoder_outputs = canidate_info["encoder_outputs"]
        if encoder_outputs is None:
            with torch.inference_mode():
                encoder_outputs = self.model.get_encoder()(
                    input_ids=input_ids, attention_mask=attention_mask
                )

        with torch.inference_mode():
            output = self.model(
                encoder_outputs=encoder_outputs,
                return_dict=True,
                decoder_input_ids=decoder_input_ids,
            )

        # Go from percentage to log probablity
        token_probs = -(f.log_softmax(output.logits[0][sequence_token_length], dim=0))
        # Using negative token_probs to get the highest probabilties because the probs are in log and in log the highest prob is the lowest number
        _, top_k_token_ids = torch.topk(-token_probs, k=num_token_options, dim=0)
        for token_id in top_k_token_ids:
            new_decoder_input_ids = decoder_input_ids.clone()
            new_decoder_input_ids[0][sequence_token_length + 1] = int(token_id)

            yield GenerationCanidate(
                current_probability + float(token_probs[int(token_id)]),
                {
                    "forced_decoder_ids": new_decoder_input_ids,
                    "sequence_token_length": sequence_token_length + 1,
                    "encoder_outputs": encoder_outputs,
                },
            )

    def generate_usage_options_prob_based(self, batch) -> list[list[dict]]:
        results = []
        total_probability = 0
        MAX_ITERATIONS = 100
        MINIMUM_PROBAILITY = -log(0.001)
        MINIMUM_TOTAL_PROBABILITY = 0.95

        generation_queue = PriorityQueue()

        generation_queue.put(
            GenerationCanidate(
                0,
                {
                    "forced_decoder_ids": torch.zeros(
                        [1, self.MAX_SEQUENCE_LENGTH + 1], dtype=torch.int32
                    ),
                    "sequence_token_length": 0,
                    "encoder_outputs": None,
                },
            )
        )

        i = 0
        while (
            not generation_queue.empty()
            and i < MAX_ITERATIONS
            and total_probability <= MINIMUM_TOTAL_PROBABILITY
        ):
            next_generations = self.__get_output_with_probs(
                batch, generation_queue.get()
            )
            i += 1

            for generation in next_generations:
                # Add to result if the eos token is reached or we already have on result and the current path probability is lower than the minimum probability
                if (
                    generation.data["forced_decoder_ids"][-1][
                        generation.data["sequence_token_length"]
                    ]
                    == 1
                    or generation.data["sequence_token_length"]
                    >= self.MAX_SEQUENCE_LENGTH
                ):
                    total_probability += exp(-generation.probability)
                    results.append(generation)
                elif generation.probability < MINIMUM_PROBAILITY:
                    generation_queue.put(generation)

        decoded_results = self.tokenizer.batch_decode(
            [result.data["forced_decoder_ids"][0] for result in results],
            skip_special_tokens=True,
        )
        decoded_results_with_probs = [
            [
                {
                    "probability": exp(
                        -float(result.probability)
                    ),  # casting current_probability to float because it is a tensor
                    "sequence_token_length": result.data["sequence_token_length"],
                    "usageOptions": self.format_usage_options(output),
                }
                for result, output in zip(results, decoded_results)
            ]
        ]
        return decoded_results_with_probs

    def generate_usage_options(self, batch: dict[str, list]) -> list[list[dict]]:
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
            batch_size = 512

        dataloader, _ = reviews.get_dataloader(
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
