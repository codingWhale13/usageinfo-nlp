import asyncio
import functools
import json
import itertools
import random
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from statistics import mean, quantiles, variance
from typing import Callable, ItemsView, Iterable, Iterator, Optional, Union

from numpy import mean, var
import helpers.label_selection as ls
from evaluation.scoring import DEFAULT_METRICS
from evaluation.scoring.evaluation_cache import EvaluationCache
from helpers.review import Review
from helpers.worker import Worker
import data_augmentation.core as da_core


class ReviewSet:
    """A ReviewSet object holds a set of reviews and makes them easily accessible.

    Data can be loaded from and saved to JSON files in the appropriate format.
    """

    latest_version = 4

    def __init__(
        self, version: str, reviews: dict, save_path: Optional[str] = None
    ) -> "ReviewSet":
        """load data and make sure it is structured according to our latest JSON format"""
        self.version = version
        self.reviews = reviews

        self.validate()

        self.save_path = save_path

    def __len__(self) -> int:
        return len(self.reviews)

    def __eq__(self, other: "ReviewSet") -> bool:
        if self.version != other.version:
            return False

        for review in self:
            if review not in other:
                return False

        for review in other:
            if review not in self:
                return False

        return True

    def __contains__(self, obj: Union[str, Review]) -> bool:
        if isinstance(obj, Review):
            obj = obj.review_id
        return obj in self.reviews

    def __iter__(self) -> Iterator[Review]:
        yield from self.reviews.values()

    def __or__(self, other: "ReviewSet") -> "ReviewSet":
        return self.merge(other, allow_new_reviews=True, inplace=False)

    def __ior__(self, other: "ReviewSet") -> "ReviewSet":
        self.merge(other, allow_new_reviews=True, inplace=True)
        return self

    def __copy__(self):
        return self.from_reviews(*self)

    def __deepcopy__(self, memo):
        return self.from_reviews(*deepcopy(list(self), memo))

    def __str__(self) -> str:
        reviews = "{\n" + ",\n".join([str(review) for review in self]) + "}"
        return f"ReviewSet version {self.version}, reviews: {reviews}"

    def __getitem__(self, review_id: str) -> Review:
        if isinstance(review_id, slice):
            return [self.reviews[review_id] for review_id in list(self.reviews)][
                review_id
            ]
        else:
            return self.reviews[review_id]

    def __delitem__(self, review_id: str):
        del self.reviews[review_id]

    def __setitem__(self, review_id: str, review: Review):
        self.reviews[review_id] = review

    @classmethod
    def from_dict(cls, data: dict, save_path: Optional[str] = None) -> "ReviewSet":
        version = data.get("version", 0)
        if version < cls.latest_version:
            if version < 1:
                exit(
                    1,
                    "Automatic upgrade from version 0 is not supported.\nPlease resort to the manual upgrade process.",
                )
            print(
                f"Auto-upgrading your json review set to version {cls.latest_version} for usage (current version: {data.get('version')})...\nThis will not override your file unless you save this reviewset!"
            )
            from helpers.upgrade_json_files import upgrade_to_latest_version

            data = upgrade_to_latest_version(data)

        reviews = {
            review_id: Review(review_id=review_id, data=review_data)
            for review_id, review_data in data.get("reviews", {}).items()
        }

        return cls(data.get("version"), reviews, save_path)

    @classmethod
    def from_reviews(
        cls, *reviews: Review, save_path: Optional[str] = None
    ) -> "ReviewSet":
        return cls(
            cls.latest_version,
            {review.review_id: review for review in reviews},
            save_path,
        )

    @classmethod
    def from_files(
        cls, *source_paths: Union[str, Path], save_path: Optional[str] = None
    ) -> "ReviewSet":
        if len(source_paths) == 0:
            raise ValueError("Expected at least one source path argument")

        def get_review_set(path: Union[str, Path]):
            with open(path) as file:
                return cls.from_dict(json.load(file))

        review_sets = []

        for path in source_paths:
            absolute_path = Path(path).expanduser().resolve()
            if absolute_path.is_dir():
                for file in absolute_path.glob("*.json"):
                    review_sets.append(get_review_set(str(file)))
            elif absolute_path.is_file():
                review_sets.append(get_review_set(str(absolute_path)))

        review_set = functools.reduce(
            lambda review_set_1, review_set_2: review_set_1 | review_set_2, review_sets
        )

        review_set.save_path = (
            str(Path(source_paths[0]).expanduser().resolve())
            if len(source_paths) == 1
            else save_path
        )

        return review_set

    def items(self) -> ItemsView[str, Review]:
        return self.reviews.items()

    def add(self, review: Review, add_new=True) -> None:
        if review in self:
            self[review.review_id] |= review
        elif add_new:
            self.reviews[review.review_id] = review

    def get_review(self, review_id: str) -> Review:
        return self.reviews[review_id]

    def count_common_reviews(self, other: "ReviewSet") -> int:
        common_review_counter = 0
        for review in other:
            if review in self:
                common_review_counter += 1
        return common_review_counter

    def count_new_reviews(self, other: "ReviewSet") -> int:
        return len(other) - self.count_common_reviews(other)

    def get_all_label_ids(self) -> set:
        label_ids = set()
        for review in self:
            label_ids |= review.get_label_ids()
        return label_ids

    def remove_label(self, label_id: str, inplace=True) -> Optional["ReviewSet"]:
        review_set = self if inplace else deepcopy(self)
        for review in review_set:
            review.remove_label(label_id, inplace=True)

        if not inplace:
            return review_set

    def get_usage_options(self, label_id: str) -> list:
        usage_options = list(
            itertools.chain(*[review.get_usage_options(label_id) for review in self])
        )
        if not usage_options:
            raise ValueError(f"Label {label_id} not found in any review")
        return usage_options

    async def __async_score(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ):
        scoring_queue = asyncio.Queue()

        def review_score(review):
            return review.score(label_id, reference_label_id, metric_ids)

        for review in self.reviews_with_labels({label_id, reference_label_id}):
            scoring_queue.put_nowait(partial(review_score, review))

        worker = Worker(scoring_queue)
        await worker.run()

    def score(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ):
        metric_ids_openai = set(
            metric_id for metric_id in metric_ids if "openai" in metric_id
        )
        metric_ids = set(metric_ids).difference(metric_ids_openai)

        if len(metric_ids_openai) > 0:
            asyncio.run(
                self.__async_score(label_id, reference_label_id, metric_ids_openai)
            )
        if len(metric_ids) > 0:
            for review in self.reviews_with_labels({label_id, reference_label_id}):
                review.score(label_id, reference_label_id, metric_ids)

        EvaluationCache.get().save_to_disk()  # save newly calculated scores to disk

    def get_agg_scores(
        self,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
        metric_ids: Union[set, list] = DEFAULT_METRICS,
    ) -> dict[dict[str, float]]:
        aggregations = {
            "mean": mean,
            "variance": variance,
            "quantiles (n=4)": quantiles,
        }
        scores = [
            review.get_scores(
                label_id, *reference_label_candidates, metric_ids=metric_ids
            )
            for review in self
        ]
        scores = list(filter(lambda x: x is not None, scores))
        if len(scores) < 2:
            raise ValueError(
                "At least two reviews are required to calculate aggregated scores"
            )
        agg_scores = {"num_reviews": len(scores)}
        for metric_id in metric_ids:
            agg_scores[metric_id] = {
                agg_name: agg_func([score[metric_id] for score in scores])
                for agg_name, agg_func in aggregations.items()
            }
        return agg_scores

    def reviews_with_labels(self, label_ids: set[str]) -> list[Review]:
        """Returns a review set containing only reviews with the given labels"""
        relevant_reviews = [
            review for review in self if label_ids <= review.get_label_ids()
        ]
        return self.from_reviews(*relevant_reviews)

    def validate(self) -> None:
        if self.version != self.latest_version:
            raise ValueError(
                f"only the latest format (v{self.latest_version})"
                "of our JSON format is supported"
            )

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

        merged_review_set = self if inplace else copy(self)

        for review in review_set:
            merged_review_set.add(review, add_new=allow_new_reviews)

        if not inplace:
            return merged_review_set

    def get_data(self) -> dict:
        """get data in correct format of the latest version"""
        result = {
            "version": self.version,
            "reviews": {review_id: review.data for review_id, review in self.items()},
        }
        return result

    def drop_review(
        self, obj: Union[str, Review], inplace=True
    ) -> Optional["ReviewSet"]:
        if isinstance(obj, Review):
            obj = obj.review_id

        if not inplace:
            reviews = deepcopy(self.reviews)
            reviews.pop(obj, None)
            return ReviewSet.from_dict({"version": self.version, "reviews": reviews})

        self.reviews.pop(obj, None)

    def filter(
        self, filter_function: Callable[[Review], bool], inplace=True, invert=False
    ) -> Optional["ReviewSet"]:
        reviews = self if inplace else copy(self)
        for review in copy(reviews):
            # if invert is True, we want to drop all reviews that match the filter function. Otherwise, we want to drop all reviews that do not match the filter function.
            if invert == bool(filter_function(review)):
                reviews.drop_review(review)

        if not inplace:
            return reviews

    def filter_with_label_strategy(
        self,
        selection_strategy: ls.LabelSelectionStrategyInterface,
        inplace=True,
        invert=False,
    ) -> Optional["ReviewSet"]:
        return self.filter(
            lambda review: review.get_label_from_strategy(selection_strategy),
            inplace=inplace,
            invert=invert,
        )

    def get_dataloader(
        self,
        tokenizer,
        model_max_length: int,
        for_training: bool,
        selection_strategy: ls.LabelSelectionStrategyInterface = None,
        multiple_usage_options_strategy: str = None,
        include_augmentations: bool = False,
        seed: int = None,
        prompt_id: str = "avetis_v1",
        **dataloader_args: dict,
    ):
        from torch.utils.data import DataLoader
        import random

        if seed is not None:
            random.seed(seed)
        tokenized_reviews = (
            data_point
            for data_points in (
                review.get_tokenized_datapoints(
                    selection_strategy=selection_strategy,
                    tokenizer=tokenizer,
                    max_length=model_max_length,
                    for_training=for_training,
                    multiple_usage_options_strategy=multiple_usage_options_strategy,
                    include_augmentations=include_augmentations,
                    prompt_id=prompt_id,
                )
                for review in self
            )
            for data_point in data_points
            if None
            not in data_point.values()  # remove datapoints with None values, since there was an error with the tokenization
        )

        # If selection_strategy is specified the output should not be 0 which is used as the default value
        if selection_strategy:
            tokenized_reviews = filter(lambda x: 0 not in x.values(), tokenized_reviews)

        tokenized_reviews = list(tokenized_reviews)
        random.shuffle(tokenized_reviews)
        return DataLoader(
            tokenized_reviews,
            **dataloader_args,
        )

    def split(
        self, fraction: float, seed: int = None
    ) -> tuple["ReviewSet", "ReviewSet"]:
        random.seed(seed)

        reviews = copy(list(self))
        random.shuffle(reviews)
        split_index = max(1, int(len(reviews) * fraction))

        return (
            ReviewSet.from_reviews(*reviews[:split_index]),
            ReviewSet.from_reviews(*reviews[split_index:]),
        )

    def create_dataset(
        self,
        dataset_name: str,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        test_split: float,
        contains_usage_split: Optional[float] = None,
        augmentation: da_core.ReviewAugmentation = None,
        seed: int = None,
    ) -> tuple["ReviewSet", dict]:
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

        reviews = deepcopy(
            self.filter_with_label_strategy(label_selection_strategy, inplace=False)
        )
        has_usage_options = lambda review: bool(
            review.get_label_from_strategy(label_selection_strategy)["usageOptions"]
        )
        reviews_with_usage = list(reviews.filter(has_usage_options, inplace=False))
        reviews_without_usage = list(
            reviews.filter(has_usage_options, inplace=False, invert=True)
        )

        for review in reviews:
            dataset_label = review.get_label_from_strategy(label_selection_strategy)

            for label_id, label in copy(review["labels"]).items():
                if label is not dataset_label:
                    del review["labels"][label_id]

            # when creating a dataset the datasets field in a review will only contain the dataset that is currently being created, same for test
            dataset_label["datasets"] = {dataset_name: "train"}
            dataset_label["augmentations"] = []

        dataset_length = len(reviews)
        if dataset_length == 0:
            raise ValueError("There is no review that has any of the specified labels.")

        if augmentation is not None:
            augmentation.augment(label_selection_strategy, *reviews)

        if contains_usage_split is not None:
            target_usage_split = (
                contains_usage_split * (1 - test_split) + 0.5 * test_split
            )
            target_no_usage_split = 1 - target_usage_split

            dataset_length = min(
                len(reviews_with_usage) / target_usage_split,
                len(reviews_without_usage) / target_no_usage_split,
            )

            reviews_with_usage = reduce_reviews(
                reviews_with_usage, round(dataset_length * target_usage_split)
            )
            reviews_without_usage = reduce_reviews(
                reviews_without_usage, round(dataset_length * target_no_usage_split)
            )

        test_reviews = random.sample(
            reviews_with_usage,
            round(dataset_length * (test_split * 0.5)),
        ) + random.sample(
            reviews_without_usage,
            round(dataset_length * (test_split * 0.5)),
        )

        for review in test_reviews:
            label = review.get_label_from_strategy(label_selection_strategy)
            label["datasets"] = {dataset_name: "test"}

        dataset_length = len(self)

        return reviews, {
            "num_test_reviews": len(test_reviews),
            "num_train_reviews": dataset_length - len(test_reviews),
            "test_split": round(len(test_reviews) / dataset_length, 3),
            "train_usage_split": round(
                len(set(reviews_with_usage) - set(test_reviews))
                / (dataset_length - len(test_reviews)),
                3,
            ),
            "test_usage_split": round(
                len(test_reviews)
                and len(set(reviews_with_usage) & set(test_reviews))
                / len(test_reviews),
                3,
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

    def score_labels_pairwise(
        self, label_ids: list[str] = None, metric_ids: list[str] = DEFAULT_METRICS
    ):
        if label_ids is None:
            label_ids = self.get_all_label_ids()

        for label_id in label_ids:
            for label_id2 in label_ids:
                if label_id != label_id2:
                    self.score(label_id, label_id2, metric_ids=metric_ids)

    def compute_label_variance(
        self,
        label_ids_to_compare: Union[str, list[str]] = "all",
        variance_type: str = "reviews",
        metric_ids: list[str] = DEFAULT_METRICS,
    ):
        """Computes the variance of the pairwise scores of the labels in the review set."""
        result = {metric_id: {} for metric_id in metric_ids}

        if label_ids_to_compare == "all":
            label_ids_to_compare = self.get_all_label_ids()
        else:
            assert set(label_ids_to_compare).issubset(set(self.get_all_label_ids()))

        self.score_labels_pairwise(
            label_ids=label_ids_to_compare, metric_ids=metric_ids
        )

        if variance_type == "reviews":
            res = {metric_id: [] for metric_id in metric_ids}

            for review in self:
                pairwise_scores_per_review = {}
                for label_id, label in review.get_labels().items():
                    if label_id not in label_ids_to_compare:
                        continue

                    for ref_id, score_dict in label["scores"].items():
                        if ref_id not in label_ids_to_compare or label_id == ref_id:
                            continue

                        key = tuple(sorted([label_id, ref_id]))
                        if key not in pairwise_scores_per_review:
                            pairwise_scores_per_review[key] = {
                                metric_id: score_dict[metric_id]
                                for metric_id in metric_ids
                            }

                for metric_id in metric_ids:
                    if len(pairwise_scores_per_review) == 0:
                        continue

                    metric_scores = [
                        scores[metric_id]
                        for scores in pairwise_scores_per_review.values()
                    ]
                    res[metric_id].append(
                        (review.review_id, mean(metric_scores), var(metric_scores))
                    )

            for metric_id in metric_ids:
                result[metric_id]["expectation"] = mean([x[1] for x in res[metric_id]])
                result[metric_id]["variance"] = mean([x[2] for x in res[metric_id]])

            return result

        elif variance_type == "labels":
            pairwise_scores = {}
            for review in self:
                for label_id, label in review.get_labels().items():
                    if label_id not in label_ids_to_compare:
                        continue

                    for ref_id, score_dict in label["scores"].items():
                        if ref_id not in label_ids_to_compare:
                            continue

                        key = tuple(sorted([label_id, ref_id]))
                        if key not in pairwise_scores:
                            pairwise_scores[key] = {
                                metric_id: [] for metric_id in metric_ids
                            }

                        for metric_id in metric_ids:
                            pairwise_scores[key][metric_id].append(
                                score_dict[metric_id]
                            )

            for metric_id in metric_ids:
                result[metric_id]["expectation"] = mean(
                    [mean(x[metric_id]) for x in pairwise_scores.values()]
                )
                result[metric_id]["variance"] = mean(
                    [var(x[metric_id]) for x in pairwise_scores.values()]
                )

            return result

    def save_as(self, path: Union[str, Path]) -> None:
        self.save_path = path
        with open(path, "w") as file:
            json.dump(self.get_data(), file)

    def save(self) -> None:
        assert (
            self.save_path is not None
        ), "ReviewSet has no `save_path`; use 'save_as' method instead"
        with open(self.save_path, "w") as file:
            json.dump(self.get_data(), file)
