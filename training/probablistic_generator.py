import torch
import torch.nn.functional as f
from typing import Optional
from typing import Union
from helpers.review_set import ReviewSet
import os
from queue import PriorityQueue, Queue
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from math import exp, log
from training.generator import DEFAULT_GENERATION_CONFIG, Generator
from tqdm import tqdm
import numpy as np
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
    def __init__(
        self,
        artifact_name: Optional[str] = None,
        generation_config: str = DEFAULT_GENERATION_CONFIG,
        checkpoint: Optional[Union[int, str]] = None,
        prompt_id="original",
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_sequence_length: int = 20,
        batch_size: int = 64,
        max_iterations: int = 100,
        minimum_probability: float = 0.001,
        minimum_total_probability: float = 0.95,
        token_top_k: int = 5
    ) -> None:
        print("Initialising BatchProbabilisticGenerator with batch_size:", batch_size)
        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.BATCH_SIZE = batch_size
        self.MAX_ITERATIONS = max_iterations
        self.MINIMUM_PROBABILITY = -log(minimum_probability)
        self.MINIMUM_TOTAL_PROBABILITY = minimum_total_probability
        self.TOKEN_TOP_K = token_top_k
        self.prompt_id = prompt_id
        if artifact_name:
            super().__init__(
                artifact_name, generation_config, checkpoint, prompt_id=prompt_id
            )
        elif model and tokenizer:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
            self.model.eval()
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                "You must either supply the artifact_name or the model and tokenizer"
            )

    def __get_output_with_probs(
        self, generation_canidates: list[GenerationCanidate]
    ) -> list[tuple[float, dict]]:
        decoder_input_ids = []
        encoder_hidden_states = []
        encoder_attention_mask = []

        for x in generation_canidates:
            decoder_input_ids.append(x.data["forced_decoder_ids"])
            encoder_hidden_states.append(x.data["encoder_outputs"])
            encoder_attention_mask.append(x.data["encoder_attention_mask"])
        
        decoder_input_ids = torch.stack(decoder_input_ids).to(self.device)
        encoder_hidden_states = torch.stack(encoder_hidden_states).to(self.device)
        encoder_attention_mask = torch.stack(encoder_attention_mask).to(self.device)

        with torch.inference_mode():
            decoder_outputs = self.model.get_decoder()(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                input_ids=decoder_input_ids,
                return_dict=True
            )

            logits = self.model.lm_head(decoder_outputs[0])

        # Go from percentage to log probablity
        token_probs = -f.log_softmax(logits, dim=-1)
        # Using negative token_probs to get the highest probabilties because the probs are in log and in log the highest prob is the lowest number
        _, top_k_token_ids = torch.topk(-token_probs, k=self.TOKEN_TOP_K, dim=-1)
        for x, generation_canidate, i in zip(
            top_k_token_ids, generation_canidates, range(len(generation_canidates))
        ):
            sequence_token_length = generation_canidate.data["sequence_token_length"]
            current_probability = generation_canidate.probability
            for token_id in x[sequence_token_length]:
                new_decoder_input_ids = generation_canidate.data[
                    "forced_decoder_ids"
                ].cpu().clone()
                new_decoder_input_ids[sequence_token_length + 1] = int(token_id)
                yield GenerationCanidate(
                    current_probability
                    + float(token_probs[i][sequence_token_length][int(token_id)]),
                    {
                        "forced_decoder_ids": new_decoder_input_ids,
                        "sequence_token_length": sequence_token_length + 1,
                        "encoder_outputs": generation_canidate.data["encoder_outputs"],
                        "review_id": generation_canidate.data["review_id"],
                        "encoder_attention_mask": generation_canidate.data["encoder_attention_mask"]
                    },
                )

    def generate_usage_options_prob_based_batch(
        self, review_set: ReviewSet, decode_results=True
    ) -> dict[str, list[dict]]:
        input_queue = Queue()
        for review in review_set:
            tokenized_input = self.tokenizer(
                review.get_prompt(prompt_id=self.prompt_id),
                padding="max_length",
                max_length=512,
                truncation=False,
                return_tensors="pt",
            )
            if len(tokenized_input["input_ids"][0]) <= 512:
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

        progres_bar = tqdm(total=input_queue.qsize())
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
                            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                        )
                        for encoder_output, x in zip(
                            encoder_output_batch.last_hidden_state, encoder_input
                        ):
                            encoder_queue.put(
                                {
                                    "review_id": x["review_id"],
                                    "encoder_outputs": encoder_output,
                                    "encoder_attention_mask": x["attention_mask"],
                                    "forced_decoder_ids": torch.zeros(
                                        [self.MAX_SEQUENCE_LENGTH + 1],
                                        dtype=torch.int32,
                                    ),
                                    "sequence_token_length": 0,
                                    "input_ids": x["input_ids"],
                                }
                            )
                    input_ids = None
                    attention_mask = None

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
                elif generation.probability < self.MINIMUM_PROBABILITY:
                    generation_queue.put(generation)

            items_to_delete = []
            for review_id, review in decoder_queue.items():
                if (
                    review["iteration"] > self.MAX_ITERATIONS
                    or review["total_probability"] >= self.MINIMUM_TOTAL_PROBABILITY
                    or review["generation_queue"].empty()
                ):
                    items_to_delete.append(review_id)
                    # Free up cuda memory
                    for generation_result in review["results"]:
                        del generation_result.data["encoder_outputs"]

                    results.append(review)

            for key in items_to_delete:
                del decoder_queue[key]

            progres_bar.update(len(items_to_delete))

        progres_bar.close()
        print("Finished generating. Decoding results")
        formatted_results = {}
        for review in tqdm(results):
            review_results = []

            review_probs = []
            review_decoder_ids = []
            for result in review["results"]:
                review_probs.append(result.probability)
                review_decoder_ids.append(result.data["forced_decoder_ids"])

            review_texts = (
                self.tokenizer.batch_decode(
                    review_decoder_ids, skip_special_tokens=True
                )
                if decode_results
                else [None] * len(review_decoder_ids)
            )

            review_probabilities = np.exp(-np.array(review_probs))
            for text, decoder_token_ids, probability in zip(
                review_texts, review_decoder_ids, review_probabilities
            ):
                review_results.append(
                    {"probability": probability}
                    | (
                        {
                            "usageOptions": self.format_usage_options(text),
                            "output_text": text,
                        }
                        if decode_results
                        else {"decoder_token_ids": decoder_token_ids}
                    )
                )
            formatted_results[review["review_id"]] = review_results

        return formatted_results

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        raise NotImplementedError()
