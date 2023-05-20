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


class Generator:
    def __init__(
        self,
        artifact_name,
        generation_config: str = DEFAULT_GENERATION_CONFIG,
        checkpoint: Optional[Union[int, str]] = None,
    ) -> None:
        global device

        self.config = utils.load_config_from_artifact_name(artifact_name)

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

    def format_usage_options(self, text_completion: str) -> List[str]:
        return [
            usage_option.strip()
            for usage_option in text_completion.split(", ")
            if usage_option.strip()
        ]

    def generate_usage_options(self, batch) -> None:
        # batch is Iterable containing [List of model inputs, List of labels, List of review_ids]
        review_ids = list(batch["review_id"])
        input_ids = batch["input"]["input_ids"].to(device)
        attention_mask = batch["input"]["attention_mask"].to(device)
        model_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generation_config,
            )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]

        return zip(review_ids, model_inputs, predictions)

    def _get_output_with_probs(
        self, batch, forced_decoder_ids=[], current_probability=1.0
    ):
        num_token_options = 5
        max_new_tokens = len(forced_decoder_ids) + 1

        with torch.no_grad():
            output = self.model.generate(
                **batch["input"].to(device),
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                forced_decoder_ids=forced_decoder_ids,
                length_penalty=1.5,
            )
        # print(output)

        probs = f.softmax(output["scores"][-1][0], dim=-1)
        # print(probs)
        best_token_ids = torch.argsort(probs, descending=True)[:num_token_options]
        # print(best_token_ids)
        i = 0
        cumul_prob = 0
        for token_id in best_token_ids:
            cumul_prob += probs[token_id]
            i += 1
            if cumul_prob > 0.99:
                break

        for token_id in best_token_ids[:i]:
            yield {
                "forced_decoder_ids": forced_decoder_ids
                + [(max_new_tokens, int(token_id))],
                "current_probability": current_probability * probs[token_id],
            }

    def generate_usage_options2(self, batch) -> None:
        # review_ids = list(batch["review_id"])
        input_ids = batch["input"]["input_ids"].to(device)
        # attention_mask = batch["input"]["attention_mask"].to(device)
        model_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        generation_queue = [{"forced_decoder_ids": [], "current_probability": 1.0}]
        results = []
        while generation_queue:
            next_generations = self._get_output_with_probs(
                batch=batch, **generation_queue.pop(0)
            )
            # print(list(next_generations))

            for generation in next_generations:
                if generation["forced_decoder_ids"][-1][1] == 1:
                    results.append(generation)
                    continue
                if generation["current_probability"] > 0.001:
                    generation_queue.append(generation)

        print("results:", results)
        print("generation_queu:e", generation_queue)

        decoded_results = self.tokenizer.batch_decode(
            [
                [token_id for _, token_id in result["forced_decoder_ids"]]
                for result in results
            ]
        )
        decoded_results_with_probs = [
            (float(result["current_probability"]), output)
            for result, output in zip(results, decoded_results)
        ]
        print(model_inputs)
        decoded_results_with_probs = sorted(
            decoded_results_with_probs, key=lambda x: x[0], reverse=True
        )
        for prob, output in decoded_results_with_probs:
            print(prob, output)
        sum_of_probs = sum([prob for prob, _ in decoded_results_with_probs])
        print(sum_of_probs)

        print("------------------")

        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         max_new_tokens=1,
        #         min_new_tokens=1,
        #         forced_decoder_ids=decoder_ids,
        #     )

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        if not label_id and not verbose:
            raise ValueError(
                "Specify either a label_id to save the labels or set verbose to True. (Or both)"
            )

        dataloader, _ = reviews.get_dataloader(
            batch_size=32,
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
            print(f"Label Metadata: {label_metadata}", end="\n\n")

        for batch in dataloader:
            usage_options_batch = self.generate_usage_options2(batch)
            # for review_id, model_input, usage_options in usage_options_batch:
            #     if label_id is not None:
            #         reviews[review_id].add_label(
            #             label_id=label_id,
            #             usage_options=usage_options,
            #             metadata=label_metadata,
            #         )
            #     if verbose:
            #         print(f"Review {review_id}")
            #         print(f"Model input:\n{model_input}")
            #         print(f"Usage options:\n\t{usage_options}", end="\n\n")
