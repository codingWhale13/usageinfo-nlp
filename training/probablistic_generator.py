import torch
import torch.nn.functional as f
from typing import Optional
from typing import Union
from transformers.models.t5.modeling_t5 import BaseModelOutput
from helpers.review_set import ReviewSet
import os
from queue import PriorityQueue, Queue
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from math import exp, log
from training.generator import DEFAULT_GENERATION_CONFIG, Generator

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


class BatchProbabilisticGenerator(Generator):
    MAX_SEQUENCE_LENGTH = 20
    BATCH_SIZE = 512

    def __init__(
        self,
        artifact_name: Optional[str] = None,
        generation_config: str = DEFAULT_GENERATION_CONFIG,
        checkpoint: Optional[Union[int, str]] = None,
        prompt_id="original",
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        if artifact_name:
            super().__init__(
                artifact_name, generation_config, checkpoint, prompt_id=prompt_id
            )
        elif model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                "You must either supply the artifact_name or the model and tokenizer"
            )

    def __get_output_with_probs(
        self, generation_canidates: list[GenerationCanidate]
    ) -> list[tuple[float, dict]]:
        decoder_input_ids = torch.stack(
            [x.data["forced_decoder_ids"] for x in generation_canidates]
        ).to(self.device)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=torch.stack(
                [x.data["encoder_outputs"] for x in generation_canidates]
            ).to(self.device)
        )
        num_token_options = 5

        with torch.inference_mode():
            output = self.model(
                encoder_outputs=encoder_outputs,
                return_dict=True,
                decoder_input_ids=decoder_input_ids,
            )

        # Go from percentage to log probablity
        token_probs = -f.log_softmax(output.logits, dim=-1)
        # Using negative token_probs to get the highest probabilties because the probs are in log and in log the highest prob is the lowest number
        _, top_k_token_ids = torch.topk(-token_probs, k=num_token_options, dim=-1)
        for x, generation_canidate, i in zip(
            top_k_token_ids, generation_canidates, range(len(generation_canidates))
        ):
            sequence_token_length = generation_canidate.data["sequence_token_length"]
            current_probability = generation_canidate.probability
            for token_id in x[sequence_token_length]:
                new_decoder_input_ids = generation_canidate.data[
                    "forced_decoder_ids"
                ].clone()
                new_decoder_input_ids[sequence_token_length + 1] = int(token_id)
                yield GenerationCanidate(
                    current_probability
                    + float(token_probs[i][sequence_token_length][int(token_id)]),
                    {
                        "forced_decoder_ids": new_decoder_input_ids,
                        "sequence_token_length": sequence_token_length + 1,
                        "encoder_outputs": generation_canidate.data["encoder_outputs"],
                        "input_ids": generation_canidate.data["input_ids"],
                        "attention_mask": generation_canidate.data["input_ids"],
                        "review_id": generation_canidate.data["review_id"],
                    },
                )

    def generate_usage_options_prob_based_batch(
        self, review_set: ReviewSet
    ) -> dict[str, list[dict]]:
        input_queue = Queue()
        for review in review_set:
            tokenized_input = self.tokenizer(
                review.get_prompt(),
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            input_queue.put(
                {
                    "review_id": review.review_id,
                    "input_ids": tokenized_input["input_ids"][0],
                    "attention_mask": tokenized_input["attention_mask"][0],
                }
            )

        encoder_queue = Queue(maxsize=self.BATCH_SIZE * 2)
        decoder_queue = {}
        results = []

        while not input_queue.empty() or len(decoder_queue) > 0:
            if len(decoder_queue) < self.BATCH_SIZE and not input_queue.empty():
                encoder_input = []
                if encoder_queue.qsize() < self.BATCH_SIZE:
                    for _ in range(self.BATCH_SIZE):
                        if input_queue.empty():
                            break
                        encoder_input.append(input_queue.get())

                    input_ids = torch.stack([x["input_ids"] for x in encoder_input]).to(
                        self.device
                    )
                    attention_mask = torch.stack(
                        [x["attention_mask"] for x in encoder_input]
                    ).to(self.device)
                    with torch.inference_mode():
                        encoder_output_batch = self.model.get_encoder()(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        for encoder_output, x in zip(
                            encoder_output_batch.last_hidden_state, encoder_input
                        ):
                            encoder_queue.put(
                                {
                                    "review_id": x["review_id"],
                                    "encoder_outputs": encoder_output,
                                    "forced_decoder_ids": torch.zeros(
                                        [self.MAX_SEQUENCE_LENGTH + 1],
                                        dtype=torch.int32,
                                    ),
                                    "sequence_token_length": 0,
                                    "input_ids": x["input_ids"],
                                }
                            )

            while not encoder_queue.empty() and len(decoder_queue) < self.BATCH_SIZE:
                generation_queue = PriorityQueue()
                generation_canidate_data = encoder_queue.get()
                generation_queue.put(GenerationCanidate(0, generation_canidate_data))
                decoder_queue[generation_canidate_data["review_id"]] = {
                    "generation_queue": generation_queue,
                    "total_probability": 0,
                    "results": [],
                    "iteration": 0,
                    "review_id": generation_canidate_data["review_id"],
                }

            generation_canidates = []
            for generation_queue in decoder_queue.values():
                generation_canidates.append(generation_queue["generation_queue"].get())

            next_generations = self.__get_output_with_probs(generation_canidates)

            MAX_ITERATIONS = 100
            MINIMUM_PROBAILITY = -log(0.001)
            MINIMUM_TOTAL_PROBABILITY = 0.95

            for generation in next_generations:
                review_id = generation.data["review_id"]
                current_review = decoder_queue[review_id]
                generation_queue = current_review["generation_queue"]
                current_review["iteration"] += 1
                if (
                    generation.data["forced_decoder_ids"][
                        generation.data["sequence_token_length"]
                    ]
                    == 1
                    or generation.data["sequence_token_length"]
                    >= self.MAX_SEQUENCE_LENGTH
                ):
                    current_review["total_probability"] += exp(-generation.probability)
                    current_review["results"].append(generation)
                elif generation.probability < MINIMUM_PROBAILITY:
                    generation_queue.put(generation)

            items_to_delete = []
            for review_id, review in decoder_queue.items():
                if (
                    review["iteration"] > MAX_ITERATIONS
                    or review["total_probability"] >= MINIMUM_TOTAL_PROBABILITY
                    or review["generation_queue"].empty()
                ):
                    items_to_delete.append(review_id)
                    results.append(review)

            for key in items_to_delete:
                del decoder_queue[key]

        formatted_results = {}
        for review in results:
            review_results = []
            for result in review["results"]:
                prob = exp(-result.probability)
                text = self.tokenizer.decode(
                    result.data["forced_decoder_ids"], skip_special_tokens=True
                )

                review_results.append(
                    {
                        "probability": prob,
                        "sequence_token_length": result.data["sequence_token_length"],
                        "usageOptions": self.format_usage_options(text),
                    }
                )
            formatted_results[review["review_id"]] = review_results

        return formatted_results

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        raise NotImplementedError()
