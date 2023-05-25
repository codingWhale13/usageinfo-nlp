import torch
import torch.nn.functional as f
from typing import List, Optional

from training import utils
from helpers.review_set import ReviewSet


class Generator:
    def __init__(
        self,
        artifact_name,
        generation_config: dict,
        checkpoint: Optional[int] = None,
        output_probabilities: str = "best",  # can be "all", "best" or None
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}
        checkpoint = torch.load(utils.get_model_path(self.model_artifact))
        model_config = utils.get_model_config_from_checkpoint(
            checkpoint["model"], checkpoint
        )

        self.model, self.tokenizer, self.max_length = model_config
        self.model.to(self.device)
        self.model.eval()
        self.generation_config = generation_config
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

        res = [None for _ in range(len(predictions))]

        if self.output_probabilities == "best":
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
                [
                    float(torch.prod(token_probs[i]))
                    for i in range(len(predicted_tokens))
                ],
                [len(predicted_tokens[i]) for i in range(len(predicted_tokens))],
                [
                    float(a ** (1 / b))
                    for a, b in zip(
                        [
                            torch.prod(token_probs[i])
                            for i in range(len(predicted_tokens))
                        ],
                        [
                            len(predicted_tokens[i])
                            for i in range(len(predicted_tokens))
                        ],
                    )
                ],
                [
                    self.tokenizer.decode(self.get_elements_until_zero(review[1:]))
                    for review in outputs["sequences"]
                ],
            )

            res = [
                [
                    {
                        "probability": x[0],
                        "token_length": x[1],
                        "geometric_mean_probability": x[2],
                        "prediction": x[3],
                    }
                ]
                for x in res
            ]

        return zip(review_ids, model_inputs, predictions, res)

    def _get_output_with_probs(
        self, batch, forced_decoder_ids=[], current_probability=1.0, token_length=0
    ):
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
                length_penalty=1.5,
            )

        probs = f.softmax(output["scores"][-1][0], dim=0)

        best_token_ids = torch.argsort(probs, descending=True)[:num_token_options]

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
                "token_length": token_length + 1,
            }

    def generate_usage_options_prob_based(self, batch) -> None:
        review_ids = list(batch["review_id"])
        input_ids = batch["input"]["input_ids"].to(self.device)
        attention_mask = batch["input"]["attention_mask"].to(self.device)

        model_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        generation_queue = [
            {"forced_decoder_ids": [], "current_probability": 1.0, "token_length": 0}
        ]
        results = []
        while generation_queue:
            next_generations = self._get_output_with_probs(
                batch=batch, **generation_queue.pop(0)
            )

            for generation in next_generations:
                if generation["forced_decoder_ids"][-1][1] == 1:
                    results.append(generation)
                    continue
                if generation["current_probability"] > 0.001:
                    generation_queue.append(generation)

        # ensure that normally generated result is also in results

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                **self.generation_config,
            )

        scores = outputs.scores

        normalized_scores = [f.softmax(score, dim=1) for score in scores]

        probs_for_generated = torch.FloatTensor(
            [
                normalized_scores[token_number][
                    0,
                    token,  # here 0 because the beams are sorted by the probability of the whole sequence, so the first beam is the most probable one
                ]
                for token_number, token in enumerate(outputs.sequences[0][1:])
            ]
        )
        generated = self.tokenizer.decode(outputs.sequences[0][1:])

        predictions = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]

        decoded_results = self.tokenizer.batch_decode(
            [
                [token_id for _, token_id in result["forced_decoder_ids"]]
                for result in results
            ]
        )

        decoded_results_with_probs = [
            (
                float(
                    result["current_probability"]
                ),  # casting current_probability to float because it is a tensor
                result["token_length"],
                float(result["current_probability"]) ** (1 / result["token_length"]),
                output,
            )
            for result, output in zip(results, decoded_results)
        ]  # is a list with tuples (probability, length, geometric_mean_probability, prediction)

        # solution which is generated by the model should be in the list
        if generated not in [x[3] for x in decoded_results_with_probs]:
            decoded_results_with_probs.append(
                (
                    float(torch.prod(probs_for_generated)),
                    len(probs_for_generated),
                    float(
                        torch.prod(probs_for_generated)
                        ** (1 / len(probs_for_generated))
                    ),
                    generated,
                )
            )

        decoded_results_with_probs = [
            {
                "probability": x[0],
                "token_length": x[1],
                "geometric_mean_probability": x[2],
                "prediction": x[3],
            }
            for x in sorted(
                decoded_results_with_probs, key=lambda x: x[0], reverse=True
            )
        ]

        return zip(review_ids, model_inputs, predictions, [decoded_results_with_probs])

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
