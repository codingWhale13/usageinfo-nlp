from copy import deepcopy
import functools
from pathlib import Path
from typing import Union, Optional, Iterator
import json
import random
from statistics import mean, variance, quantiles

from helpers.review import Review


class ReviewSet:
    """A ReviewSet object holds a set of reviews and makes them easily accessible.

    Data can be loaded from and saved to JSON files in the appropriate format.
    """

    latest_version = 3

    def __init__(
        self, version: str, reviews: dict, source_path: Optional[str] = None
    ) -> "ReviewSet":
        """load data and make sure it is structured according to our latest JSON format"""
        self.version = version
        self.validate_version()

        self.reviews = reviews
        self.validate_reviews()

        self.source_path = source_path

    def __len__(self) -> int:
        return len(self.reviews)

    def __contains__(self, obj: Union[str, Review]) -> bool:
        if isinstance(obj, Review):
            obj = obj.review_id
        return obj in self.reviews

    def __iter__(self) -> Iterator[Review]:
        yield from self.reviews.values()

    def __getitem__(self, review_id: str) -> Review:
        return self.get_review(review_id)

    def __or__(self, other: "ReviewSet") -> "ReviewSet":
        return self.merge(other, allow_new_reviews=True, inplace=False)

    def __ior__(self, other: "ReviewSet") -> None:
        return self.merge(other, allow_new_reviews=True, inplace=True)

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewSet":
        reviews = {
            review_id: Review(review_id=review_id, data=review_data)
            for review_id, review_data in data.get("reviews", {}).items()
        }
        return cls(data.get("version"), reviews)

    @classmethod
    def from_reviews(cls, *reviews: Review) -> "ReviewSet":
        return cls(cls.latest_version, {review.review_id: review for review in reviews})

    @classmethod
    def from_files(cls, *source_paths: Union[str, Path]) -> "ReviewSet":
        if len(source_paths) == 0:
            raise ValueError("Expected at least one source path argument")

        def get_review_set(path: Union[str, Path]):
            with open(path) as file:
                return cls.from_dict(json.load(file))

        review_sets = (get_review_set(path) for path in source_paths)

        return functools.reduce(
            lambda review_set_1, review_set_2: review_set_1 | review_set_2, review_sets
        )

    def items(self):
        return self.reviews.items()

    def get_review(self, review_id: str) -> Review:
        return self.reviews[review_id]

    def get_all_label_ids(self) -> set:
        label_ids = set()
        for review in self:
            label_ids |= review.get_label_ids()
        return label_ids

    def get_scores(self, label_id: str, reference_label_id: str, metrics=None):
        if metrics is None:
            from evaluation.scoring import DEFAULT_METRICS

            metrics = DEFAULT_METRICS

        individual_scores = [
            review.get_scores(label_id, reference_label_id, metrics=metrics)
            for review in self.reviews_with({label_id, reference_label_id})
        ]
        aggregated_scores = {}
        available_metrics = individual_scores[0].keys()

        aggregations = {
            "mean": mean,
            "variance": variance,
            "quantiles (n=4)": quantiles,
        }

        for metric in available_metrics:
            aggregated_scores[metric] = {}
            for aggregation_name in aggregations:
                aggregated_scores[metric][aggregation_name] = aggregations[
                    aggregation_name
                ]([scores[metric] for scores in individual_scores])
        return aggregated_scores

    def reviews_with(self, label_ids: set):
        relevant_reviews = [
            review for review in self if label_ids <= review.get_label_ids()
        ]
        return self.from_reviews(*relevant_reviews)

    def __len__(self) -> int:
        return len(self.reviews)

    def validate_version(self) -> None:
        if self.version != self.latest_version:
            raise ValueError(
                f"only the latest format (v{self.latest_version})"
                "of our JSON format is supported"
            )

    def validate_reviews(self) -> None:
        for review in self:
            review.validate()

    def merge(
        self,
        review_set: "ReviewSet",
        allow_new_reviews: bool = False,
        inplace=False,
    ) -> Optional["ReviewSet"]:
        """Merges foreign ReviewSet into this ReviewSet

        Args:
            review_set (ReviewSet): foreign ReviewSet to merge into this one
            allow_new_reviews (bool, optional): if set to True, all unseen reviews from `review_set` will be added. Defaults to False.
            inplace (bool, optional): if set to True, overwrites object data; otherwise, creates a new ReviewSet object. Defaults to False.
        """

        assert (
            self.version == review_set.version == self.latest_version
        ), f"expected ReviewSets in latest format (v{self.latest_version})"

        existing_reviews = deepcopy(self.reviews)
        additional_reviews = deepcopy(review_set.reviews)

        for review_id, review in additional_reviews.items():
            if review_id in existing_reviews:
                existing_reviews[
                    review_id
                ] |= review  # merge labels of existing reviews
            elif allow_new_reviews:
                existing_reviews[review_id] = review  # add new reviews

        if not inplace:
            return ReviewSet.from_reviews(*existing_reviews.values())

        self.reviews = existing_reviews

    def get_data(self) -> dict:
        """get data in correct format of the latest version"""
        result = {
            "version": self.version,
            "reviews": {review_id: review.data for review_id, review in self.items()},
        }
        return result

    def drop_review(self, obj: Union[str, Review], inplace=True) -> Optional[Review]:
        if isinstance(obj, Review):
            obj = obj.review_id

        if not inplace:
            reviews = deepcopy(self.reviews)
            reviews.pop(obj, None)
            return ReviewSet.from_dict({"version": self.version, "reviews": reviews})

        self.reviews.pop(obj, None)

    def create_dataset(
        self,
        dataset_name: str,
        label_id: str,
        test_split: float,
        contains_usage_split: Optional[float] = None,
        seed: int = None,
    ) -> dict:
        def reduce_reviews(reviews: list[Review], target_num: int):
            if len(reviews) < target_num:
                raise ValueError(
                    f"Can't reduce list with length {len(reviews)} to {target_num}"
                )
            random.shuffle(reviews)
            for review in reviews[target_num:]:
                self.drop_review(review)
            return reviews[:target_num]

        random.seed(seed)

        contains_usage = []
        contains_no_usage = []

        for review in self:
            try:
                label = review.get_label(label_id)
                label["datasets"][dataset_name] = "train"
                if len(label["usageOptions"]) == 0:
                    contains_no_usage.append(review)
                else:
                    contains_usage.append(review)
            except KeyError:
                self.drop_review(review)

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

        test_reviews = random.sample(
            contains_usage,
            round(dataset_length * (test_split * 0.5)),
        ) + random.sample(
            contains_no_usage,
            round(dataset_length * (test_split * 0.5)),
        )

        for review in test_reviews:
            label = review.get_label(label_id)
            label["datasets"][dataset_name] = "test"

        dataset_length = len(self)
        return {
            "num_test_reviews": len(test_reviews),
            "num_train_reviews": dataset_length - len(test_reviews),
            "test_split": round(len(test_reviews) / dataset_length, 3),
            "train_usage_split": round(
                len(set(contains_usage) - set(test_reviews))
                / (dataset_length - len(test_reviews)),
                3,
            ),
            "test_usage_split": round(
                len(set(contains_usage) & set(test_reviews)) / len(test_reviews), 3
            ),
        }

    def get_dataset(self, dataset_name: str) -> tuple[dict, dict]:
        train_data = {}
        test_data = {}
        data = self.reviews.copy()
        for review_id, review in data.items():
            label = review.get_label_for_dataset(dataset_name)
            if label is None:
                continue

            data_point = {
                "product_title": review["product_title"],
                "review_body": review["review_body"],
                "usage_options": label["usageOptions"],
            }
            if label["datasets"][dataset_name] == "train":
                train_data[review_id] = data_point
            elif label["datasets"][dataset_name] == "test":
                test_data[review_id] = data_point
            else:
                raise ValueError(
                    f"Unknown dataset type {label['datasets'][dataset_name]}"
                )

        return train_data, test_data

    def save(self) -> None:
        assert (
            self.source_path is not None
        ), "ReviewSet has no source path; use 'save_as' method instead"
        with open(self.source_path, "w") as file:
            json.dump(self.get_data(), file)

    def save_as(self, path: Union[str, Path]) -> None:
        self.source_path = path
        with open(path, "w") as file:
            json.dump(self.get_data(), file)
