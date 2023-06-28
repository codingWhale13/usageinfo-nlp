import torch
import torch.nn.functional as f
from typing import List, Optional
from typing import Union

from training import utils
from helpers.review_set import ReviewSet
from training.utils import get_config
import os
from queue import PriorityQueue, Queue

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


class ProbabilisticGenerator(Generator):
    MAX_SEQUENCE_LENGTH = 20
    BATCH_SIZE = 128

    def __init__(self, artifact_name, generation_config: str = ..., checkpoint: int | str | None = None, output_probabilities: str = "best") -> None:
        super().__init__(artifact_name, generation_config, checkpoint, output_probabilities)

    def __get_output_with_probs(
        self, generation_canidates: list[GenerationCanidate]
    ) -> list[tuple[float, dict]]:
        decoder_input_ids = torch.stack(
            x.data["forced_decoder_ids"] for x in generation_canidates
        ).to(self.device)
        input_ids = torch.stack(x.data["input_ids"] for x in generation_canidates).to(
            self.device
        )
        attention_mask = torch.stack(
            x.data["attention_mask"] for x in generation_canidates
        ).to(self.device)
        encoder_outputs = torch.stack(
            x.data["encoder_outputs"] for x in generation_canidates
        ).to(self.device)
        num_token_options = 5

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                decoder_input_ids=decoder_input_ids,
            )

        # Go from percentage to log probablity
        token_probs = -f.log_softmax(output.logits, dim=-1)
        # Using negative token_probs to get the highest probabilties because the probs are in log and in log the highest prob is the lowest number
        _, top_k_token_ids = torch.topk(-token_probs, k=num_token_options, dim=-1)
        for x, generation_canidate in zip(top_k_token_ids, generation_canidates):
            sequence_token_length = generation_canidate.data["sequence_token_length"]
            current_probability = generation_canidate.probability
            for token_id in x:
                new_decoder_input_ids = generation_canidate.data["decoder_input_ids"]
                new_decoder_input_ids[0][sequence_token_length + 1] = int(token_id)

                yield GenerationCanidate(
                    current_probability + float(token_probs[int(token_id)]),
                    {
                        "forced_decoder_ids": new_decoder_input_ids,
                        "sequence_token_length": sequence_token_length + 1,
                        "encoder_outputs": generation_canidate.data["encoder_outputs"],
                        "input_ids": generation_canidate.data["input_ids"],
                        "attention_mask": generation_canidate.data["input_ids"],
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
                    "input_ids": batch["input"]["input_ids"],
                    "attention_mask": batch["input"]["attention_mask"],
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

    def generate_usage_options_prob_based_batch(self, review_set: ReviewSet) -> list[dict]:
        input_queue = Queue() 
        for review in review_set:
            input_queue.put(
                {
                    "review_id": review.review_id,
                    ** self.tokenizer(review.get_prompt(), padding="max_length", max_length=512, truncation=True)
                }
            )
      
        encoder_queue = Queue(maxsize=self.BATCH_SIZE*2)
        decoder_queue = []
        results = []

        while not input_queue.empty():
            if len(decoder_queue) < self.BATCH_SIZE:
                if len(encoder_queue) < self.BATCH_SIZE:
                    encoder_input = []
                    for _ in range(self.BATCH_SIZE):
                        if input_queue.empty():
                            break
                        encoder_input.append(input_queue.get())

                    
                    input_ids = torch.stack([x["input_ids"] for x in encoder_input]).to(self.device)

                    with torch.inference_mode():
                        encoder_output_batch = self.model.get_encoder(
                            input_ids=input_ids
                        )
                        for encoder_output in zip(encoder_output_batch, encoder_input):
                            encoder_queue.put({
                                "review_id": encoder_input["review_id"],
                                "encoder_output": encoder_output
                            })
                    

            while not encoder_queue.empty() and len(decoder_queue) < self.BATCH_SIZE:
                generation_queue = PriorityQueue()
                generation_queue.put(
                    GenerationCanidate(
                        0,
                        encoder_queue.get()
                    )
                )
                decoder_queue.append({
                    "generation_queue": generation_queue,
                    "total_probability": 0,
                    "results": [],
                    "iteration": 0
                })


            generation_canidates = []
            for generation_queue in decoder_queue:
                generation_canidates.append(generation_queue.get())
            
            next_generations = self.__get_output_with_probs(generation_canidates)

            MAX_ITERATIONS = 100
            MINIMUM_PROBAILITY = -log(0.001)
            MINIMUM_TOTAL_PROBABILITY = 0.95
        

            for i, generation in enumerate(next_generations):
                # Add to result if the eos token is reached or we already have on result and the current path probability is lower than the minimum probability

                current_review = decoder_queue[i]
                generation_queue = current_review["generation_queue"]
                current_review["iteration"] += 1
                if (
                    generation.data["forced_decoder_ids"][-1][
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

                if current_review["iteration"] > MAX_ITERATIONS or current_review["total_probability"] >= MINIMUM_TOTAL_PROBABILITY:
                    decoder_queue.remove(current_review)
                    results.append(current_review)

        return results

            
            



            


    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        self.generate_usage_options_prob_based_batch([review.get_prompt() for review in reviews])