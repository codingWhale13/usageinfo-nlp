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
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from heapq import *

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
        token_top_k: int = 5,
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
            print(self.model)

        else:
            raise ValueError(
                "You must either supply the artifact_name or the model and tokenizer"
            )
        print(
            "parameters",
            self.MAX_SEQUENCE_LENGTH,
            self.MAX_ITERATIONS,
            self.MINIMUM_PROBABILITY,
            self.MINIMUM_TOTAL_PROBABILITY,
        )
        print("prompt:", self.prompt_id)

    def __get_output_with_probs(
        self, generation_canidates: list[GenerationCanidate], decoder_queues: dict
    ) -> list[tuple[float, dict]]:
        decoder_input_ids = []
        encoder_hidden_states = []
        encoder_attention_mask = []

        for x in generation_canidates:
            decoder_input_ids.append(x.data["forced_decoder_ids"])
            decoder_queue = decoder_queues[x.data["review_id"]]
            encoder_hidden_states.append(decoder_queue["encoder_outputs"])
            encoder_attention_mask.append(decoder_queue["encoder_attention_mask"])

        decoder_input_ids = pad_sequence(
            decoder_input_ids, padding_value=0, batch_first=True
        ).to(self.device)

        encoder_hidden_states = torch.stack(encoder_hidden_states).to(self.device)
        encoder_attention_mask = torch.stack(encoder_attention_mask).to(self.device)

        with torch.inference_mode():
            decoder_outputs = self.model.get_decoder()(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                input_ids=decoder_input_ids,
                return_dict=True,
            )
            self.decoder_computation_steps += 1
            logits = self.model.lm_head(decoder_outputs[0])  # .detach()

        # Go from percentage to log probablity
        token_probs = -f.log_softmax(logits, dim=-1)
        # Using negative token_probs to get the highest probabilties because the probs are in log and in log the highest prob is the lowest number
        _, top_k_token_ids = torch.topk(-token_probs, k=self.TOKEN_TOP_K, dim=-1)
        new_generation_canidates = []
        for x, generation_canidate, i in zip(
            top_k_token_ids, generation_canidates, range(len(generation_canidates))
        ):
            sequence_token_length = generation_canidate.data["sequence_token_length"]
            new_sequence_token_length = sequence_token_length + 1
            review_id = generation_canidate.data["review_id"]
            decoder_queues[review_id]["iteration"] += 1
            current_probability = generation_canidate.probability

            for token_id in x[sequence_token_length]:
                new_generation_canidates.append(
                    GenerationCanidate(
                        current_probability
                        + float(token_probs[i][sequence_token_length][int(token_id)]),
                        {
                            "forced_decoder_ids": torch.cat(
                                [
                                    generation_canidate.data["forced_decoder_ids"],
                                    torch.tensor([token_id]),
                                ],
                                dim=-1,
                            ),
                            "sequence_token_length": new_sequence_token_length,
                            "review_id": review_id,
                            # "encoder_outputs": generation_canidate.data["encoder_outputs"],
                            # "encoder_attention_mask" : generation_canidate.data["encoder_attention_mask"]
                        },
                    )
                )

        return new_generation_canidates

    def generate_label(self, reviews: ReviewSet, label_id: str) -> None:
        reviews = reviews.filter(
            lambda review: label_id not in review.get_label_ids(), inplace=False
        )
        if len(reviews) == 0:
            print(
                f"All reviews already labelled with label_id: {label_id}. Skipping generating"
            )
            return None

        for review_id, results in self.generate_usage_options_prob_based_batch(
            reviews, cluster_results=True
        ).items():
            usage_options = results[0]["usageOptions"]
            reviews.get_review(review_id).add_label(
                label_id, usage_options, metadata={"probabilistic_generations": results}
            )

    def generate_usage_options_prob_based_batch(
        self, review_set: ReviewSet, cluster_results=True
    ) -> dict[str, list[dict]]:
        input_queue = Queue()
        self.decoder_computation_steps = 0
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
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )

                        last_hidden_state = (
                            encoder_output_batch.last_hidden_state
                        )  # .detach()
                        # del encoder_output_batch.last_hidden_state

                        for encoder_output, x in zip(last_hidden_state, encoder_input):
                            encoder_queue.put(
                                {
                                    "review_id": x["review_id"],
                                    "encoder_outputs": encoder_output,
                                    "encoder_attention_mask": x["attention_mask"],
                                    "forced_decoder_ids": torch.zeros(
                                        [1],
                                        dtype=torch.int32,
                                    ),
                                    "sequence_token_length": 0,
                                    "input_ids": x["input_ids"],
                                }
                            )
                        del input_ids
                        del attention_mask
                        del encoder_input
                        del encoder_output_batch

            while not encoder_queue.empty() and len(decoder_queue) < self.BATCH_SIZE:
                generation_queue = []
                generation_canidate_data = encoder_queue.get()
                encoder_outputs = generation_canidate_data["encoder_outputs"]
                encoder_attention_mask = generation_canidate_data[
                    "encoder_attention_mask"
                ]

                heappush(
                    generation_queue, GenerationCanidate(0, generation_canidate_data)
                )
                decoder_queue[generation_canidate_data["review_id"]] = {
                    "generation_queue": generation_queue,
                    "total_probability": 0,
                    "results": [],
                    "iteration": 0,
                    "review_id": generation_canidate_data["review_id"],
                    "encoder_outputs": encoder_outputs,
                    "encoder_attention_mask": encoder_attention_mask,
                }

                del generation_canidate_data["encoder_outputs"]
                del generation_canidate_data["encoder_attention_mask"]
                del generation_canidate_data

            generation_canidates = []
            for generation_queue in decoder_queue.values():
                generation_canidates.append(
                    heappop(generation_queue["generation_queue"])
                )

            next_generations = self.__get_output_with_probs(
                generation_canidates, decoder_queue
            )

            for generation in next_generations:
                review_id = generation.data["review_id"]
                current_review = decoder_queue[review_id]
                generation_queue = current_review["generation_queue"]
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
                elif (
                    generation.probability < self.MINIMUM_PROBABILITY
                    or len(current_review["results"]) == 0
                ):
                    heappush(generation_queue, generation)

            # del next_generations
            items_to_delete = []
            for review_id, review in decoder_queue.items():
                if (
                    (
                        review["iteration"] > self.MAX_ITERATIONS
                        and len(review["results"]) > 0
                    )
                    or review["total_probability"] >= self.MINIMUM_TOTAL_PROBABILITY
                    or len(review["generation_queue"]) == 0
                ):
                    items_to_delete.append(review_id)
                    results.append(review)

            for key in items_to_delete:
                del decoder_queue[key]["encoder_outputs"]
                del decoder_queue[key]["encoder_attention_mask"]
                del decoder_queue[key]

            # torch.cuda.empty_cache()
            """
            print(
                "Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB"
            )
            print(
                "Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 2), "GB"
            )
            """
            # print(torch.cuda.memory_stats("cuda"))

            progres_bar.update(len(items_to_delete))

        progres_bar.close()
        print("Finished generating. Decoding results")
        formatted_results = {}
        simple_format_results = {}
        for review in tqdm(results, desc="Accumalting raw results"):
            total_usage_options = len(review["results"])
            decoder_results = torch.zeros(
                total_usage_options, self.MAX_SEQUENCE_LENGTH + 1, dtype=torch.int32
            )
            probabilities = torch.empty(total_usage_options, dtype=torch.float32)
            for i, result in enumerate(review["results"]):
                decoder_results[i][
                    : len(result.data["forced_decoder_ids"])
                ] = result.data["forced_decoder_ids"]
                probabilities[i] = result.probability

            probabilities = torch.exp(-probabilities)
            simple_format_results[review["review_id"]] = [
                probabilities,
                decoder_results,
                None,
                None,
            ]

        embeddings = None
        if cluster_results:
            import time

            start_time = time.perf_counter()
            all_decoder_token_ids = torch.concat(
                [review[1] for review in simple_format_results.values()]
            )
            decoded_texts = self.tokenizer.batch_decode(
                all_decoder_token_ids, skip_special_tokens=True
            )
            print("Decoded results in:", time.perf_counter() - start_time)

            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cuda",
            )
            embeddings = model.encode(
                decoded_texts,
                show_progress_bar=True,
                batch_size=256,
                convert_to_numpy=True,
            )

            results_keys = list(simple_format_results.keys())

            from sklearn.cluster import AgglomerativeClustering

            i = 0
            for j_key in tqdm(
                range(len(results_keys)), desc="Clustering decoded results"
            ):
                review_texts = []
                num_texts_for_review = len(
                    simple_format_results[results_keys[j_key]][1]
                )
                simple_format_results[results_keys[j_key]][2] = decoded_texts[
                    i : i + num_texts_for_review
                ]

                clustering_labels = [0]
                if num_texts_for_review > 1:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        metric="cosine",
                        linkage="complete",
                        distance_threshold=0.2,
                    ).fit(embeddings[i : i + num_texts_for_review])
                    clustering_labels = clustering.labels_
                simple_format_results[results_keys[j_key]][3] = clustering_labels
                i += num_texts_for_review

        final_format_results = {}
        for review_id, review in tqdm(
            simple_format_results.items(), desc="Formatting results"
        ):
            final_format_results[review_id] = []
            if cluster_results:
                for probability, decoder_token_ids, text, cluster in zip(
                    review[0], review[1], review[2], review[3]
                ):
                    final_format_results[review_id].append(
                        {
                            "probability": float(probability),
                            "usageOptions": self.format_usage_options(text),
                            "decoder_token_ids": decoder_token_ids.tolist(),
                            "cluster": cluster,
                        }
                    )
            else:
                for probability, decoder_token_ids in zip(
                    review[0],
                    review[1],
                ):
                    final_format_results[review_id].append(
                        {
                            "probability": float(probability),
                            "decoder_token_ids": decoder_token_ids,
                        }
                    )
        return final_format_results

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
            formatted_results[review["review_id"]] = {
                "results": review_results,
                "iterations": review["iteration"],
            }

        return formatted_results
