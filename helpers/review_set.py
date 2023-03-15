from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List
import json
import random

REVIEW_ATTRIBUTES = [
    "marketplace",
    "customer_id",
    "product_id",
    "product_parent",
    "product_title",
    "product_category",
    "star_rating",
    "helpful_votes",
    "total_votes",
    "vine",
    "verified_purchase",
    "review_headline",
    "review_body",
    "review_date",
    "labels",
]

LABEL_ATTRIBUTES = [
    "datasets",
    "createdAt",
    "metadata",
    "scores",
    "usageOptions",
]


class ReviewSet:
    newest_version = 3

    def __init__(self, data: dict, source_path: Optional[str] = None):
        self.version = data.get("version")
        self.reviews = data["reviews"]

        valid, error_msg = self.is_valid()
        if not valid:
            raise ValueError(f"JSON is invalid: {error_msg}")

        self.source_path = source_path

    @classmethod
    def from_files(cls, *source_paths: Union[str, Path]):
        source_paths = list(source_paths)
        with open(source_paths.pop(0)) as file:
            reviews = cls(json.load(file))

        for path in source_paths:
            with open(path) as file:
                reviews2 = cls(json.load(file))
            reviews.merge(reviews2, allow_new_reviews=True, inplace=True)

        return reviews

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data)

    def get_review(self, review_id: str) -> dict:
        return self.reviews[review_id]

    def get_labels(self, review_id: str) -> dict:
        return self.get_review(review_id)["labels"]

    def get_label(self, review_id: str, label_id: str) -> dict:
        return self.get_labels(review_id)[label_id]

    def is_valid(self) -> bool:
        """determine if data is tecorrectly structured"""

        if self.version != self.newest_version:
            return (
                False,
                f"only the newest version ({self.newest_version}) of our JSON format is supported",
            )

        for review_id, review in self.reviews.items():
            if sorted(review.keys()) != sorted(REVIEW_ATTRIBUTES):
                return (
                    False,
                    f"wrong keys for review '{review_id}': {sorted(review.keys())}\n{sorted(REVIEW_ATTRIBUTES)}",
                )

            if not isinstance(review["labels"], dict):
                return False, "field 'labels' is not dict"
            for label_id, label in review["labels"].items():
                if not isinstance(label, dict):
                    return False, f"label '{label_id}' is not dict"
                if sorted(label.keys()) != sorted(LABEL_ATTRIBUTES):
                    return (
                        False,
                        f"wrong keys for label '{label_id}' in review '{review_id}': {sorted(label.keys())} \n {sorted(LABEL_ATTRIBUTES)}",
                    )
                if not isinstance(label["usageOptions"], list):
                    return (
                        False,
                        f"usage options of '{label_id}' in review '{review_id}' is not a list but {type(label['usageOptions'])}",
                    )
                if not isinstance(label["metadata"], dict):
                    return (
                        False,
                        f"metadata of '{label_id}' in review '{review_id}' is not a dict but {type(label['metadata'])}",
                    )
                if not isinstance(label["scores"], dict):
                    return (
                        False,
                        f"scores of '{label_id}' in review '{review_id}' is not a dict but {type(label['scores'])}",
                    )
                if not isinstance(label["datasets"], dict):
                    return (
                        False,
                        f"datasets of '{label_id}' in review '{review_id}' is not a dict but {type(label['datasets'])}",
                    )
                try:
                    datetime.fromisoformat(label["createdAt"])
                except Exception as e:
                    return (
                        False,
                        f"createdAt timestamp of '{label_id}' in review '{review_id}' is not ISO 8601",
                    )

        return True, ""

    def merge(
        self,
        review_set: "ReviewSet",
        allow_new_reviews: bool = False,
        inplace=False,
    ):
        """Merges foreign ReviewSet into this ReviewSet

        Args:
            review_set (ReviewSet): foreign ReviewSet to merge into this one
            allow_new_reviews (bool, optional): if set to True, all unseen reviews from `review_set` will be added. Defaults to False.
            inplace (bool, optional): if set to True, overwrites object data; otherwise, creates a new ReviewSet object. Defaults to False.
        """

        assert (
            self.version == review_set.version == self.newest_version
        ), "expected ReviewSets in newest format"

        our_reviews = deepcopy(self.reviews)
        foreign_reviews = deepcopy(review_set.reviews)

        # add labels and metadata that is missing in this object's data
        for review_id, our_review in our_reviews.items():
            if review_id in foreign_reviews:
                for label_id, foreign_label in foreign_reviews[review_id][
                    "labels"
                ].items():
                    if label_id not in our_review["labels"]:
                        our_review["labels"][label_id] = foreign_label
                    else:
                        our_label = our_review["labels"][label_id]
                        # validate same usage options
                        assert (
                            our_label["usageOptions"] == foreign_label["usageOptions"]
                        ), f"'{label_id}' in '{review_id}' has inconsistent usage options"

                        # merge scores
                        for ref_id, foreign_score_set in foreign_label[
                            "scores"
                        ].items():
                            if ref_id not in our_label["scores"]:
                                our_label["scores"][ref_id] = foreign_score_set
                            else:
                                our_scores = our_label["scores"][ref_id]
                                our_scores = set(our_scores + foreign_score_set)

                        # merge datasets
                        datasets = our_label["datasets"]
                        datasets = set(datasets + foreign_label["datasets"])

                        # merge metadata
                        our_metadata = our_label["metadata"]
                        for metadata_id, foreign_metadatum in foreign_label[
                            "metadata"
                        ].items():
                            if metadata_id not in our_metadata:
                                our_metadata[metadata_id] = foreign_metadatum

        if allow_new_reviews:
            our_review_ids = our_reviews.keys()
            for review_id, foreign_review in foreign_reviews.items():
                if review_id not in our_review_ids:
                    our_reviews[review_id] = foreign_review

        if inplace:
            self.reviews = our_reviews
            return self
        else:
            data = {
                "version": self.version,
                "reviews": our_reviews,
            }
            return self.from_dict(data)

    def add_label(
        self, review_id: str, label_id: str, usage_options: list[str]
    ) -> None:
        """Add a new label to a review

        Args:
            review_id (str): where we want to add the label
            label_id (str): origin of the label
            usage_options (list[str]): labelled data
        """

        assert review_id in self.reviews, f"review '{review_id}' not found"
        assert label_id not in self.get_labels(
            review_id
        ), f"label '{label_id}' already exists in review '{review_id}'"

        self.reviews[review_id]["labels"][label_id] = {
            "createdAt": datetime.now().astimezone().isoformat(),  # using ISO 8601
            "usageOptions": usage_options,
            "scores": {},
            "datasets": {},
            "metadata": {},
        }

    def get_data(self) -> dict:
        """get data in correct format of the newest version"""
        result = {
            "version": self.version,
            "reviews": self.reviews,
        }
        return result

    def drop_review(self, id):
        return self.reviews.pop(id, None)

    def create_dataset(
        self, dataset_name, label_id, test_split, contains_usage_split=None, seed=None
    ):
        def reduce_reviews(l: List[str], target_num: int):
            if len(l) < target_num:
                raise ValueError(
                    f"Can't reduce list with length {len(l)} to {target_num}"
                )
            random.shuffle(l)
            for id in l[target_num:]:
                self.drop_review(id)
            return l[:target_num]

        random.seed(seed)

        contains_usage = []
        contains_no_usage = []

        for id, _ in self.reviews.items():
            try:
                label = self.get_label(id, label_id)
                label["datasets"][dataset_name] = "train"
                if len(label["usageOptions"]) == 0:
                    contains_no_usage.append(id)
                else:
                    contains_usage.append(id)
            except KeyError:
                self.drop_review(id)

        dataset_length = len(contains_usage) + len(contains_no_usage)

        if contains_usage_split is not None:
            target_usage_split = (
                contains_usage_split * (1 - test_split) + 0.5 * test_split
            )
            target_no_usage_split = 1 - target_usage_split

            dataset_length = min(
                len(contains_usage) / target_usage_split,
                len(contains_no_usage) / target_no_usage_split,
            )

            contains_usage = reduce_reviews(
                contains_usage, round(dataset_length * target_usage_split)
            )
            contains_no_usage = reduce_reviews(
                contains_no_usage, round(dataset_length * target_no_usage_split)
            )

        test_ids = random.sample(
            contains_usage,
            round(dataset_length * (test_split * 0.5)),
        ) + random.sample(
            contains_no_usage,
            round(dataset_length * (test_split * 0.5)),
        )

        for id in test_ids:
            label = self.get_label(id, label_id)
            label["datasets"][dataset_name] = "test"

        dataset_length = len(self.reviews)
        return {
            "num_test_reviews": len(test_ids),
            "num_train_reviews": dataset_length - len(test_ids),
            "test_split": round(len(test_ids) / dataset_length, 3),
            "train_usage_split": round(
                len(set(contains_usage) - set(test_ids))
                / (dataset_length - len(test_ids)),
                3,
            ),
            "test_usage_split": round(
                len(set(contains_usage) & set(test_ids)) / len(test_ids), 3
            ),
        }

    def get_dataset(self, dataset_name):
        def get_label_for_dataset(review_id, dataset_name):
            for id, label in self.get_labels(review_id=review_id).items():
                if dataset_name in label["datasets"]:
                    return label
            return None

        train_data = {}
        test_data = {}
        data = self.reviews.copy()
        for id, review in data.items():
            label = get_label_for_dataset(id, dataset_name)
            if label is None:
                continue

            data_point = {
                "product_title": review["product_title"],
                "review_body": review["review_body"],
                "usage_options": label["usageOptions"],
            }
            if label["datasets"][dataset_name] == "train":
                train_data[id] = data_point
            elif label["datasets"][dataset_name] == "test":
                test_data[id] = data_point
            else:
                raise ValueError(
                    f"Unknown dataset type {label['datasets'][dataset_name]}"
                )

        return train_data, test_data

    def save(self) -> None:
        assert (
            self.source_path is not None
        ), "ReviewSet has no source path; use save_as method instead"
        with open(self.source_path, "w") as file:
            json.dump(self.get_data(), file)

    def save_as(self, path: Union[str, Path]) -> None:
        with open(path, "w") as file:
            json.dump(self.get_data(), file)
