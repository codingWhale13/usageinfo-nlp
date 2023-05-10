import itertools
import random
from copy import copy, deepcopy
from datetime import datetime, timezone
from typing import Iterable, Optional, Union
import json

import dateutil.parser

import helpers.label_selection as ls
from evaluation.scoring import DEFAULT_METRICS


class Review:
    review_attributes = {
        "customer_id",
        "helpful_votes",
        "labels",
        "marketplace",
        "product_category",
        "product_id",
        "product_parent",
        "product_title",
        "review_body",
        "review_date",
        "review_headline",
        "star_rating",
        "total_votes",
        "verified_purchase",
        "vine",
    }

    label_attributes = {
        "createdAt",
        "datasets",
        "metadata",
        "scores",
        "usageOptions",
        "augmentations",
    }

    def __init__(self, review_id: str, data: dict) -> None:
        self.review_id = review_id
        self.data = data

    def __getitem__(self, key: str) -> Union[str, int, dict]:
        if key in self.data:
            return self.data[key]

        raise ValueError(f"review '{self.review_id}' does not contain key '{key}'")

    def __eq__(self, other) -> bool:
        if isinstance(other, Review):
            return self.__key() == other.__key()
        else:
            return False

    def __key(self) -> str:
        return self.review_id

    def __hash__(self) -> int:
        return hash(self.__key())

    def __or__(self, other: "Review") -> "Review":
        return self.merge_labels(other, inplace=False)

    def __ior__(self, other: "Review") -> None:
        self.merge_labels(other, inplace=True)
        return self

    def __copy__(self):
        return Review(self.review_id, copy(self.data))

    def __deepcopy__(self, memo):
        return Review(self.review_id, deepcopy(self.data, memo))

    def __str__(self):
        simple_data = {}
        simple_data["product_title"] = self.data["product_title"]
        simple_data["review_headline"] = self.data["review_headline"]
        simple_data["review_body"] = self.data["review_body"]
        simple_data["labels"] = self.data["labels"].copy()
        for label_id, label in simple_data["labels"].items():
            simple_data["labels"][label_id] = label["usageOptions"]

        return f"Review {self.review_id} " + json.dumps(simple_data, indent=4)

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def get_labels(self) -> dict:
        return self.data.get("labels", {})

    def get_label_ids(self) -> set[str]:
        return set(self.get_labels().keys())

    def get_usage_options(self, label_id: str) -> list[str]:
        return self.get_label_for_id(label_id).get("usageOptions", [])

    def get_label_from_strategy(self, strategy) -> Optional[dict]:
        if not isinstance(strategy, ls.LabelSelectionStrategyInterface):
            raise ValueError(
                f"strategy '{type(strategy)}' doesn't implement LabelSelectionStrategyInterface"
            )
        return strategy.retrieve_label(self)

    def get_label_for_dataset(
        self, *dataset_names: Union[str, tuple[str, str]]
    ) -> Optional[dict]:
        return self.get_label_from_strategy(ls.DatasetSelectionStrategy(*dataset_names))

    def get_label_for_id(self, *label_ids: str) -> Optional[dict]:
        return self.get_label_from_strategy(ls.LabelIDSelectionStrategy(*label_ids))

    def add_label(
        self,
        label_id: str,
        usage_options: list[str],
        metadata: dict = {},
        overwrite: bool = False,
    ) -> None:
        if not overwrite:
            assert (
                label_id not in self.get_labels()
            ), f"label '{label_id}' already exists in review '{self.review_id}'"

        self.data["labels"][label_id] = {
            # using ISO 8601 with UTC timezone, https://stackoverflow.com/a/63731605
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "usageOptions": usage_options,
            "scores": {},
            "datasets": {},
            "metadata": metadata,
        }

    def tokenize(
        self,
        tokenizer,
        text: str,
        for_training: bool,
        is_input: bool,
        max_length: int = float("inf"),
    ) -> Optional[dict]:
        tokens = tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=not for_training
        )

        # Remove batch dimension, since we only have one example
        tokens["input_ids"] = tokens["input_ids"][0]
        tokens["attention_mask"] = tokens["attention_mask"][0]

        if not is_input and for_training:
            ids = tokens["input_ids"]
            # You need to set the pad tokens for the input to -100 for some Transformers (https://github.com/huggingface/transformers/issues/9770)>
            tokens["input_ids"][ids[:] == tokenizer.pad_token_id] = -100

        return tokens if len(tokens["input_ids"]) <= max_length else None

    def _get_output_texts_from_strategy(
        self, usage_options: list[str], strategy: str = None
    ) -> list[str]:
        if not strategy or strategy == "default":
            return [", ".join(usage_options)]
        elif strategy == "flat":
            return usage_options or [""]
        elif strategy.startswith("shuffle"):
            usage_options = copy(usage_options)
            random.shuffle(usage_options)
            if not strategy.startswith("shuffle-"):
                return [", ".join(usage_options)]

            permutation_limit = strategy.split("-")[1]
            if permutation_limit == "all":
                permutation_limit = None
            else:
                try:
                    permutation_limit = int(permutation_limit)
                    if permutation_limit < 1:
                        raise ValueError(
                            "Number of permutations must be greater than 0"
                        )
                except ValueError as e:
                    if str(e) == "Number of permutations must be greater than 0":
                        raise e
                    raise ValueError(
                        f"Could not parse number of permutations for shuffle strategy '{strategy}'",
                        "Please use 'shuffle-<number_of_permutations>' or 'shuffle-all'",
                    )

            permutations = [
                ", ".join(permutation)
                for permutation in itertools.islice(
                    itertools.permutations(usage_options), permutation_limit
                )
            ]
            return permutations
        else:
            raise ValueError(f"strategy '{strategy}' not supported")

    def get_tokenized_datapoints(
        self,
        selection_strategy: ls.LabelSelectionStrategyInterface = None,
        multiple_usage_options_strategy: str = None,
        for_training: bool = False,
        **tokenization_kwargs,
    ) -> Iterable[dict]:
        def get_prompt(product_title: str, review_body: str) -> str:
            return f"Product title: {product_title} \nReview body: {review_body}\n"

        def format_dict(model_input, output, review_id) -> dict:
            return {"input": model_input, "output": output, "review_id": review_id}

        model_input = get_prompt(self["product_title"], self["review_body"])
        model_input = self.tokenize(
            text=model_input,
            is_input=True,
            for_training=for_training,
            **tokenization_kwargs,
        )

        # Returns 0 if when no selection strategy. We are using 0 instead of None because of the dataloader
        label = (
            self.get_label_from_strategy(selection_strategy)
            if selection_strategy
            else None
        )
        if not label:
            yield format_dict(model_input, 0, self.review_id)
            return

        augmentations = [(model_input, label["usageOptions"])]
        if for_training:
            for augmentation in label.get("augmentations", []):
                model_input = get_prompt(
                    augmentation.get("product_title") or self["product_title"],
                    augmentation.get("review_body") or self["review_body"],
                )
                model_input = self.tokenize(
                    text=model_input,
                    is_input=True,
                    for_training=for_training,
                    **tokenization_kwargs,
                )
                augmentations.append(
                    (
                        model_input,
                        augmentation.get("usageOptions") or label["usageOptions"],
                    )
                )
        for model_input, usage_options in augmentations:
            output_texts = self._get_output_texts_from_strategy(
                usage_options, strategy=multiple_usage_options_strategy
            )
            for output_text in output_texts:
                yield format_dict(
                    model_input,
                    self.tokenize(
                        text=output_text,
                        is_input=False,
                        for_training=for_training,
                        **tokenization_kwargs,
                    ),
                    self.review_id,
                )

    def remove_label(self, label_id: str, inplace=True) -> Optional["Review"]:
        review_without_label = (
            self if inplace else Review(self.review_id, deepcopy(self.data))
        )
        review_without_label.data["labels"].pop(label_id, None)

        if not inplace:
            return review_without_label

    def score(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ) -> None:
        """score specified metrics if not done already"""
        scores = self.get_label_for_id(label_id)["scores"]  # use reference from here on

        if reference_label_id not in scores:
            scores[reference_label_id] = {}

        available_metrics = scores.get(reference_label_id, {})
        missing_metric_ids = set(metric_ids).difference(set(available_metrics.keys()))

        if len(missing_metric_ids) > 0:
            # calculate missing metrics
            from evaluation.scoring.metrics import SingleReviewMetrics

            new_metrics = SingleReviewMetrics.from_labels(
                self.get_labels(), label_id, reference_label_id
            ).calculate(missing_metric_ids)

            for metric_id, metric_value in new_metrics.items():
                scores[reference_label_id][metric_id] = metric_value

    def get_scores(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ) -> dict[str, float]:
        """return specified scores (and calculate them internally, if missing)"""
        self.score(label_id, reference_label_id, metric_ids)
        return {
            m_id: self.get_label_for_id(label_id)["scores"][reference_label_id][m_id]
            for m_id in metric_ids
        }

    def merge_labels(
        self, other_review: "Review", inplace: bool = False
    ) -> Optional["Review"]:
        """Merge labels from another review into this one.

        This method is used to merge labels of the same review into this object.
        """
        assert self == other_review, "cannot merge labels of different reviews"
        existing_labels = self.get_labels()
        if not inplace:
            existing_labels = deepcopy(existing_labels)
        additional_labels = deepcopy(other_review.get_labels())

        for label_id, other_label in additional_labels.items():
            if label_id not in existing_labels:
                existing_labels[label_id] = other_label
            else:
                own_label = existing_labels[label_id]
                # validate same usage options
                assert (
                    own_label["usageOptions"] == other_label["usageOptions"]
                ), f"'{label_id}' in review '{other_review.review_id}' has inconsistent usage options"

                # merge scores
                for ref_id, other_score_dict in other_label["scores"].items():
                    if ref_id not in own_label["scores"]:
                        own_label["scores"][ref_id] = other_score_dict
                    else:
                        # merge different score metrics for same reference
                        own_label["scores"][ref_id].update(other_score_dict)

                # merge datasets and metadata
                own_label["datasets"] |= other_label["datasets"]
                own_label["metadata"] |= other_label["metadata"]

        if not inplace:
            return Review(
                self.review_id, self.data.copy() | {"labels": existing_labels}
            )

        self.data["labels"] = existing_labels

    def validate(self) -> None:
        error_msg_prefix = f"encountered error in review '{self.review_id}':"

        data_keys_set = set(self.data.keys())
        if not set(self.review_attributes).issubset(set(data_keys_set)):
            raise ValueError(
                f"{error_msg_prefix} wrong attribute names\n"
                f"got: {data_keys_set}\nexpected: {self.review_attributes}"
            )

        labels = self.get_labels()
        if not isinstance(labels, dict):
            raise ValueError(
                f"{error_msg_prefix} 'labels' is not of type dict but {type(labels)}"
            )

        for label_id, label in labels.items():
            if not isinstance(label, dict):
                raise ValueError(
                    f"{error_msg_prefix} label '{label_id}' is not of type dict but {type(label)}"
                )
            label_keys_set = set(label.keys())
            if label_keys_set != self.label_attributes:
                raise ValueError(
                    f"{error_msg_prefix} wrong keys in label '{label_id}'\n"
                    f"got: {label_keys_set}\nexpected: {self.label_attributes}",
                )
            if not isinstance(label["usageOptions"], list):
                raise ValueError(
                    f"{error_msg_prefix} 'usageOptions' in label '{label_id}' is not of type list but {type(label['usageOptions'])}",
                )
            if not isinstance(label["metadata"], dict):
                raise ValueError(
                    f"{error_msg_prefix} 'metadata' in label '{label_id}' is not of type dict but {type(label['metadata'])}",
                )
            if not isinstance(label["scores"], dict):
                raise ValueError(
                    f"{error_msg_prefix} 'scores' in label '{label_id}' is not of type dict but {type(label['scores'])}",
                )
            if not isinstance(label["datasets"], dict):
                raise ValueError(
                    f"{error_msg_prefix} 'datasets' in label '{label_id}' is not of type dict but {type(label['datasets'])}",
                )
            try:
                dateutil.parser.isoparse(label["createdAt"])
            except Exception:
                raise ValueError(
                    f"{error_msg_prefix} 'createdAt' timestamp in label '{label_id}' is not ISO 8601",
                )
