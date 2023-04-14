import torch
import torch.nn.functional as F
from typing import List
from typing import Union
import numpy as np

from training import utils
from helpers.review_set import ReviewSet


class Generator:
    def __init__(
        self,
        artifact_name,
        checkpoint: int,
        generation_config: dict,
    ) -> None:
        self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}
        checkpoint = torch.load(
            utils.get_model_path(self.model_artifact), map_location=torch.device("cpu")
        )
        model_config = utils.get_model_config_from_checkpoint(
            checkpoint["model"], checkpoint
        )

        self.model, self.tokenizer, self.max_length = model_config
        self.choices_per_step = generation_config["choices_per_step"]
        del generation_config["choices_per_step"]
        self.generation_config = generation_config
        self.flat = utils.get_config_from_artifact(artifact_name)["flat"]

    def format_usage_options(self, text_completion: str) -> List[str]:
        return [
            usage_option.strip()
            for usage_option in text_completion.split(", ")
            if usage_option.strip()
        ]

    def trash(self):
        #  def generate_usage_options(self, batch) -> None:
        # #     review_ids = list(batch[2])[0]
        # #     model_inputs = self.tokenizer.batch_decode(
        # #         batch[0]["input_ids"], skip_special_tokens=True
        # #     )[0]
        # #     usage_options = []

        # #     n_steps = self.generation_config["max_new_tokens"]
        # #     choices_per_step = self.generation_config["choices_per_step"]
        # #     decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])
        # #     usage_option, predictions = self.custom_generation(
        # #         n_steps, choices_per_step, batch, decoder_input_ids
        # #     )
        # #     usage_options.append(usage_option)

        # #     predictions_keys = list(predictions.keys())

        # #     covered_prod = predictions[predictions_keys[0]]["prob"]

        # #     # Reduce n_steps because we already have one iteration
        # #     n_steps -= 1
        # #     tested = 1
        # #     while covered_prod < 70:
        # #         decoder_input_ids = torch.tensor(
        # #             [[self.tokenizer(predictions[predictions_keys[tested]]["token"]).input_ids[0]]]
        # #         )
        # #         usage_option, predictions[predictions_keys[tested]]["pred"] = self.custom_generation(
        # #             n_steps, choices_per_step, batch, decoder_input_ids
        # #         )
        # #         usage_options.append(usage_option)
        # #         covered_prod += predictions[predictions_keys[tested]]["prob"]

        #      return review_ids, model_inputs, predictions, ", ".join(usage_options)

        # # def custom_generation(
        # #     self,
        # #     n_steps,
        # #     choices_per_step,
        # #     batch,
        # #     decoder_input_ids,
        # # ) -> List:
        # #     tokens = []
        # #     dicts = []
        # #     with torch.no_grad():
        # #         for _ in range(n_steps):
        # #             prediction = self.generate_token(
        # #                 choices_per_step, batch, decoder_input_ids
        # #             )
        # #             prediction_keys = list(prediction.keys())
        # #             # Append predicted next token to input
        # #             decoder_input_ids = torch.cat(
        # #                 [decoder_input_ids, torch.tensor(prediction_keys[0])[None, None]],
        # #                 dim=-1,
        # #             )
        # #             if tokens:
        # #                 dicts.append(prediction)
        # #             else:
        # #                 predictions = prediction
        # #             tokens.append(prediction_keys[0])
        # #             if prediction_keys[0] == 1:
        # #                 usage_option = [self.tokenizer.decode(token, skip_special_tokens=True) for token in list(decoder_input_ids)][0]
        # #                 break

        # #         if dicts:
        # #             for i in reversed(range(1, len(tokens) - 1)):
        # #                 dicts[i - 1][tokens[i]]["pred"] = dicts[i]

        # #             predictions[tokens[0]]["pred"] = dicts[0]
        # #         else:
        # #             predictions = prediction
        # #         return (usage_option, predictions)

        # # def generate_token(self, choices_per_step, batch, decoder_input_ids):
        # #     prediction = dict()
        # #     output = self.model(
        # #         input_ids=batch[0]["input_ids"],
        # #         attention_mask=batch[0]["attention_mask"],
        # #         decoder_input_ids=decoder_input_ids,
        # #     )
        # #     # Select logits of the first batch and the last token and apply softmax
        # #     next_token_logits = output.logits[0, -1, :]
        # #     next_token_probs = torch.softmax(next_token_logits, dim=-1)
        # #     sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
        # #     # Store tokens with highest probabilities
        # #     for choice_idx in range(choices_per_step):
        # #         token_id = sorted_ids[choice_idx]
        # #         token_prob = next_token_probs[token_id].cpu().numpy()
        # #         prediction[token_id.item()] = {
        # #             "token": self.tokenizer.decode(token_id),
        # #             "prob": 100 * token_prob,
        # #         }
        # #     return prediction
        pass

    def generate_usage_options(self, batch) -> None:
        # batch is Iterable containing [List of model inputs, List of labels, List of review_ids]
        review_ids = list(batch["review_id"])
        model_inputs = self.tokenizer.batch_decode(
            batch["input"]["input_ids"], skip_special_tokens=True
        )

        predicted_usage_options, predictions = self.generate(batch)

        if self.flat:
            for review_number, _ in enumerate(review_ids):
                covered_prod = predictions[review_number]["0.0"]["prob"]
                tested = 1

                while covered_prod < 0.5 and tested < 5:
                    new_decoder_input_ids = self.tokenizer(
                        predictions[review_number][f"0.{tested}"]["token"],
                        return_tensors="pt",
                    )["input_ids"]
                    usage_options, pred = self.generate(
                        batch, new_decoder_input_ids, offset=1
                    )
                    predicted_usage_options[review_number].extend(
                        usage_options[review_number]
                    )
                    predictions[review_number][f"0.{tested}"]["predictions"] = pred
                    covered_prod += predictions[review_number][f"0.{tested}"]["prob"]
                    tested += 1

        # decoder_input_ids = torch.tensor([0])
        # for review_number, review_score in enumerate(token_probs):
        #     while len(review_score) > 0:
        #         print("review_score", review_score)
        #         for token_score in review_score:
        #             for token_option_score in token_score["prob"]:
        #                 covered_prod = token_option_score
        #                 if covered_prod < 0.7:
        #                     new_decoder_input_ids = torch.cat(
        #                         [
        #                             decoder_input_ids,
        #                             torch.tensor(token_score["token"][None, 1, None]),
        #                         ],
        #                         dim=-1,
        #                     )
        #                     predictions, token_probs = self.generate(
        #                         batch, new_decoder_input_ids
        #                     )
        #                     usage_options[review_number].append(predictions)
        #                     scores[review_number].append(token_probs)
        #                     covered_prod += token_option_score
        #                     print("covered_prod", covered_prod)
        #             decoder_input_ids = torch.cat(
        #                 [
        #                     decoder_input_ids,
        #                     torch.tensor(token_score["token"][None, 0, None]),
        #                 ],
        #                 dim=-1,
        #             )

        return review_ids, model_inputs, predicted_usage_options, predictions

    def generate(self, batch, decoder_input_ids=None, offset=0):
        if decoder_input_ids is not None:
            return self.get_predictions_and_token_probs(
                self.model.generate(
                    **batch["input"],
                    **self.generation_config,
                    decoder_input_ids=decoder_input_ids,
                ),
                offset,
            )
        return self.get_predictions_and_token_probs(
            self.model.generate(**batch["input"], **self.generation_config), offset
        )

    def get_predictions_and_token_probs(self, outputs, offset):
        # transition_scores = self.model.compute_transition_scores(
        #     outputs.sequences, outputs.scores, normalize_logits=True
        # )
        # transition_scores = [np.exp(score.numpy()) for score in transition_scores][0]

        predicted_usage_options = self.tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=True
        )
        # predicted_tokens = [
        #     [token for token in review[1:]]
        #     for review in outputs["sequences"]
        # ][0]
        predicted_usage_options = [
            self.format_usage_options(usage_options)
            for usage_options in predicted_usage_options
        ]

        # predicted_tokens = [{
        #     "token": token,
        #     "prob": 100 * token_prob,
        # } for token, token_prob in zip(predicted_tokens, transition_scores)]

        # Use softmax to map to actual scores
        probs = [F.softmax(scores, dim=1) for scores in outputs["scores"]]

        # token_probs has review_number many entries, for every review we go through the relevant tokens and log their probability
        best_tokens = [
            [
                torch.argsort(
                    probs[token_number][review_number, :], dim=-1, descending=True
                )[: self.choices_per_step]
                for token_number, _ in enumerate(review[1:])
            ]
            for review_number, review in enumerate(outputs["sequences"])
        ]

        best_probs = [
            [
                [
                    probs[token_number][
                        review_number, best_tokens[review_number][token_number][choice]
                    ]
                    for choice in range(self.choices_per_step)
                ]
                for token_number, _ in enumerate(review[1:])
            ]
            for review_number, review in enumerate(outputs["sequences"])
        ]

        # token_probs = [
        #             {
        #                 "token": best_probs[review_number][token_number][choice],
        #                 "prob": probs[token_number][
        #                     review_number, best_probs[review_number][token_number]
        #                 ][choice],
        #             }
        #     for review_number, review in enumerate(outputs["sequences"]) for token_number, _ in enumerate(review[1:]) for choice, _ in enumerate(best_probs[review_number][token_number])
        # ]
        # print(outputs["sequences"])
        # outputs["sequences"] = [[ 0, 100,  19,  1551, 73, 1]]
        # best_probs = [[[0,1,2,3,4], [100, 101, 102, 103, 104], [19, 20, 21, 22, 23], [1551, 1552, 1553, 1554, 1555], [73, 74, 75, 76, 77], [1, 2, 3, 4, 5]]]
        predictions = []
        for review_number, review in enumerate(outputs["sequences"]):
            prediction = {}
            for token_number in reversed(range(0, len(review) - 1)):
                pred = {}
                tokens = best_tokens[review_number][token_number]
                for choice, token in enumerate(tokens):
                    pred[f"{token_number + offset}.{choice}"] = {
                        "token": self.tokenizer.decode(token),
                        "prob": best_probs[review_number][token_number][choice].item(),
                    }

                if prediction != {}:
                    pred[f"{token_number+offset}.0"]["predictions"] = prediction
                prediction = pred
            predictions.append(prediction)

        # print(predictions)
        # print("predicted_usage_options", predicted_usage_options)
        return predicted_usage_options, predictions

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        if not label_id and not verbose:
            raise ValueError(
                "Specify either a label_id to save the labels or set verbose to True. (Or both)"
            )

        dataloader = reviews.get_dataloader(
            batch_size=1,
            num_workers=2,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            for_training=False,
            shuffle=True,
        )

        if verbose:
            print(f"Generating label {label_id}...")

        for batch in dataloader:
            (
                review_id,
                model_input,
                usage_options,
                predictions,
            ) = self.generate_usage_options(batch)
            label_metadata = {
                "generator": {
                    "artifact_name": self.model_artifact["name"],
                    "checkpoint": self.model_artifact["checkpoint"] or "last",
                    "generation_config": self.generation_config,
                },
                "predictions": predictions,
            }

            if label_id is not None:
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
