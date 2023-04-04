from copy import deepcopy
from datetime import datetime
from typing import Union, Optional

from evaluation.scoring import DEFAULT_METRICS
import helpers.label_selection as ls


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
        return self.merge_labels(other, inplace=True)

    def __copy__(self):
        return Review(self.review_id, deepcopy(self.data))

    def __str__(self):
        return f"Review '{self.review_id}'\t-> Label IDs: {list(self.data['labels'].keys())}"

    def get_labels(self) -> dict:
        return self.data.get("labels", {})

    def get_label_ids(self) -> set[str]:
        return set(self.get_labels().keys())

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
    ) -> None:
        assert (
            label_id not in self.get_labels()
        ), f"label '{label_id}' already exists in review '{self.review_id}'"

        self.data["labels"][label_id] = {
            "createdAt": datetime.now().astimezone().isoformat(),  # using ISO 8601
            "usageOptions": usage_options,
            "scores": {},
            "datasets": {},
            "metadata": metadata,
        }

    def tokenize(
        self, tokenizer, max_length: int, text: str, for_training: bool, is_input: bool
    ):
        tokens = tokenizer(text, return_tensors="pt", padding="max_length")

        # Remove batch dimension, since we only have one example
        tokens["input_ids"] = tokens["input_ids"][0]
        tokens["attention_mask"] = tokens["attention_mask"][0]

        if not is_input and for_training:
            ids = tokens["input_ids"]
            # You need to set the pad tokens for the input to -100 for some Transformers (https://github.com/huggingface/transformers/issues/9770)>
            tokens["input_ids"][ids[:] == tokenizer.pad_token_id] = -100

        return tokens if len(tokens["input_ids"]) <= max_length else None

    def get_tokenized_datapoint(self, selection_strategy=None, **tokenization_kwargs):
        model_input = f'Product title: {self["product_title"]} \nReview body: {self["review_body"]}\n'
        model_input = self.tokenize(
            text=model_input, is_input=True, **tokenization_kwargs
        )

        # Returns 0 if when no selection strategy. We are using 0 instead of None because of the dataloader
        output = 0
        if selection_strategy:
            label = ", ".join(
                self.get_label_from_strategy(selection_strategy)["usageOptions"]
            )
            output = self.tokenize(text=label, is_input=False, **tokenization_kwargs)

        return model_input, output, self.review_id

    def score(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Union[set, list] = DEFAULT_METRICS,
    ) -> None:
        """score specified metrics if not done already"""
        if "scores" not in self.get_label_for_id(label_id):
            self.get_label_for_id(label_id)[
                "scores"
            ] = {}  # sanity check for JSON v3 format

        if reference_label_id not in self.get_label_for_id(label_id)["scores"]:
            self.get_label_for_id(label_id)["scores"][reference_label_id] = {}

        available_metrics = self.get_label_for_id(label_id)["scores"].get(
            reference_label_id, {}
        )

        missing_metric_ids = set(metric_ids).difference(set(available_metrics.keys()))

        # calculate missing metrics
        from evaluation.scoring.metrics import SingleReviewMetrics

        new_metrics = SingleReviewMetrics.from_labels(
            self.get_labels(), label_id, reference_label_id
        ).calculate(missing_metric_ids)

        for metric_id, metric_value in new_metrics.items():
            self.get_label_for_id(label_id)["scores"][reference_label_id][
                metric_id
            ] = metric_value

    def get_scores(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Union[set, list] = DEFAULT_METRICS,
    ) -> dict[str, float]:
        """return specified scores (and calculate them internally, if missing)"""
        self.score(label_id, reference_label_id, metric_ids)
        return {
            metric_id: self.get_label_for_id(label_id)["scores"][reference_label_id][
                metric_id
            ]
            for metric_id in metric_ids
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

        for label_id, other_label in additional_labels:
            if label_id not in existing_labels:
                existing_labels[label_id] = other_label
            else:
                own_label = existing_labels[label_id]
                # validate same usage options
                assert (
                    own_label["usageOptions"] == other_label["usageOptions"]
                ), f"'{label_id}' in review '{other_review.review_id}' has inconsistent usage options"

                # merge scores
                for ref_id, score_set in other_label["scores"].items():
                    if ref_id not in own_label["scores"]:
                        own_label["scores"][ref_id] = score_set
                    else:
                        # merge different score metrics for same reference
                        merged_score_set = set(own_label["scores"][ref_id] + score_set)
                        own_label["scores"][ref_id] = list(merged_score_set)

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
        if data_keys_set != self.review_attributes:
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
                datetime.fromisoformat(label["createdAt"])
            except Exception:
                raise ValueError(
                    f"{error_msg_prefix} 'createdAt' timestamp in label '{label_id}' is not ISO 8601",
                )
