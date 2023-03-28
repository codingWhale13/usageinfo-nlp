from copy import deepcopy
from datetime import datetime
from typing import Union, Optional


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
        return self.review_id == other.review_id

    def __key(self):
        return self.review_id

    def __hash__(self):
        return hash(self.__key())

    def __or__(self, other) -> "Review":
        return self.merge_labels(other, inplace=False)

    def __ior__(self, other) -> None:
        return self.merge_labels(other, inplace=True)

    def get_labels(self) -> dict:
        return self.data.get("labels", {})

    def get_label_ids(self) -> set[str]:
        return set(self.get_labels().keys())

    def get_label(self, label_id: str) -> dict:
        return self.get_labels().get(label_id, {})

    def get_label_for_dataset(self, dataset_name):
        for label in self.get_labels().values():
            if dataset_name in label["datasets"]:
                return label
        return None

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

    def get_metrics(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids=None,
    ):
        if metric_ids is None:
            from evaluation.scoring import DEFAULT_METRICS

            metric_ids = DEFAULT_METRICS

        metrics = (
            self.get_label(label_id).get("scores", {}).get(reference_label_id, None)
        )
        if metrics is None:
            metrics = self.__calculate_metrics(label_id, reference_label_id, metric_ids)
        else:
            missing_metrics = set(metric_ids).difference(set(metrics.keys()))
            self.set_metrics(label_id, reference_label_id, missing_metrics)
            metrics |= self.__calculate_metrics(
                label_id, reference_label_id, missing_metrics
            )

        return metrics

    def set_metrics(
        self,
        label_id: str,
        reference_label_id: str,
        metrics: dict[str, float],
    ):
        if "scores" not in self.get_label(label_id):
            self.get_label(label_id)["scores"] = {}

        self.get_label(label_id)["scores"][reference_label_id] = (
            self.get_label(label_id)["scores"].get(reference_label_id, {}) | metrics
        )

    def __calculate_metrics(
        self,
        prediction_label_id,
        reference_label_id,
        metric_ids=None,
    ):
        if metric_ids is None:
            from evaluation.scoring import DEFAULT_METRICS

            metric_ids = DEFAULT_METRICS

        from evaluation.scoring.metrics import SingleReviewMetrics

        metrics = SingleReviewMetrics.from_labels(
            self.get_labels(), prediction_label_id, reference_label_id
        ).calculate(metric_ids)
        self.set_metrics(prediction_label_id, reference_label_id, metrics)
        return metrics

    def merge_labels(self, other_review: "Review", inplace=False) -> Optional["Review"]:
        """Merge labels from another review into this one.

        This method is used to merge labels of the same review into this object.
        """
        assert self == other_review, "cannot merge labels of different reviews"
        existing_labels = deepcopy(self.get_labels())
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
            except Exception as e:
                raise ValueError(
                    f"{error_msg_prefix} 'createdAt' timestamp in label '{label_id}' is not ISO 8601",
                )
