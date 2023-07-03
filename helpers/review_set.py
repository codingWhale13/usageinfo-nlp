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
from helpers.cmd_input import get_yes_or_no_input
import glob

from numpy import mean, var
import numpy as np
import pandas as pd
import helpers.label_selection as ls
from evaluation.scoring import DEFAULT_METRICS
from evaluation.scoring.evaluation_cache import EvaluationCache
from helpers.review import Review
from helpers.worker import Worker
import data_augmentation.core as da_core
from evaluation.scoring.h_tests import h_test
from helpers.label_selection import DatasetSelectionStrategy


class ReviewSet:
    """A ReviewSet object holds a set of reviews and makes them easily accessible.

    Data can be loaded from and saved to JSON files in the appropriate format.
    """

    latest_version = 5

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

    def __sub__(self, other: "ReviewSet") -> "ReviewSet":
        return self.set_minus(other)

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
        # enabling using the * syntax to read multiple files with same pattern
        source_paths = list(itertools.chain(*[glob.glob(path, recursive=True) for path in source_paths]))
        source_paths.sort()
        
        print(source_paths)
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

    def set_minus(self, other: "ReviewSet") -> "ReviewSet":
        return ReviewSet.from_reviews(*(set(self) - set(other)))

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

    def stratified_ttv_split(
        self,
        train_split: float = 0.7,
        test_split: float = 0.2,
        val_split: float = 0.1,
        label_ids: list[str] = ["bp-chat_gpt_correction", "chat_gpt-vanilla-baseline"],
    ):
        def get_first_label_match(labels: dict, label_ids: list[str]):
            for label_id in label_ids:
                if label_id in labels:
                    return label_id
            return None

        assert train_split + test_split + val_split == 1.0

        df = self.to_pandas()
        train_list = []
        test_list = []
        val_list = []

        df["contains_usage_options"] = df["labels"].apply(
            lambda x: True
            if len(x[get_first_label_match(x, label_ids)]["usageOptions"]) > 0
            else False
        )

        df1 = df.groupby(
            [
                "contains_usage_options",
                "product_category",
                "star_rating",
                "vine",
                "verified_purchase",
            ]
        )

        for key, item in df1:
            train, test, val = np.split(
                item,
                [
                    int(train_split * len(item)),
                    int((train_split + test_split) * len(item)),
                ],
            )

            train_list.append(train)
            test_list.append(test)
            val_list.append(val)

        final_train = pd.concat(train_list).drop(columns=["contains_usage_options"])
        final_test = pd.concat(test_list).drop(columns=["contains_usage_options"])
        final_val = pd.concat(val_list).drop(columns=["contains_usage_options"])

        final_train_dict = {
            "version": 5,
            "reviews": final_train.to_dict(orient="index"),
        }
        final_test_dict = {
            "version": 5,
            "reviews": final_test.to_dict(orient="index"),
        }
        final_val_dict = {
            "version": 5,
            "reviews": final_val.to_dict(orient="index"),
        }

        return (
            ReviewSet.from_dict(final_train_dict),
            ReviewSet.from_dict(final_test_dict),
            ReviewSet.from_dict(final_val_dict),
        )

    def get_usage_options(self, label_id: str) -> list:
        usage_options = list(
            itertools.chain(*[review.get_usage_options(*label_ids) for review in self])
        )
        if not usage_options:
            raise ValueError(f"Labels {label_ids} not found in any review")
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

    def test(
        self,
        label_id_1: Union[str, ls.LabelSelectionStrategyInterface],
        label_id_2: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
        tests: list[str] = ["ttest"],
        alternatives: list[str] = ["two-sided"],
        metric_ids: Iterable[str] = DEFAULT_METRICS,
        confidence_level: float = 0.95,  # only used for bootstrap
    ):
        import numpy as np

        tests = (
            ["ttest", "wilcoxon", "bootstrap", "permutation"]
            if "all" in tests
            else tests
        )

        # iterate over all reviews and score label_id_1 against reference_label_candidates and label_id_2 against reference_label_candidates
        scores = {
            metric_id: {label_id_1: [], label_id_2: []} for metric_id in metric_ids
        }

        for review in self:
            score_1 = review.get_scores(
                label_id_1, *reference_label_candidates, metric_ids=metric_ids
            )
            score_2 = review.get_scores(
                label_id_2, *reference_label_candidates, metric_ids=metric_ids
            )

            if score_1 is None or score_2 is None:
                continue

            for metric_id in metric_ids:
                scores[metric_id][label_id_1].append(score_1[metric_id])
                scores[metric_id][label_id_2].append(score_2[metric_id])

        test_results = {}

        for test in tests:
            for metric_id in metric_ids:
                for alternative in alternatives:
                    test_results[(test, metric_id, alternative)] = h_test(
                        test,
                        np.array(scores[metric_id][label_id_1]),
                        np.array(scores[metric_id][label_id_2]),
                        alternative=alternative,
                        confidence_level=confidence_level,
                    )

        EvaluationCache.get().save_to_disk()

        return test_results

    def generate_score_report(
        self,
        folder: str,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_ids: Union[str, ls.LabelSelectionStrategyInterface],
    ):
        from evaluation.plotting.score_report import get_score_report
        from evaluation.plotting.save_report import save_report

        save_report(*get_score_report(self, folder, label_id, *reference_label_ids))

    def reviews_with_labels(self, label_ids: set[str]) -> list[Review]:
        """Returns a review set containing only reviews with the given labels"""
        relevant_reviews = [
            review for review in self if label_ids <= review.get_label_ids()
        ]
        return self.from_reviews(*relevant_reviews)

    def correct_chatGPT(self, pattern: str) -> None:
        n = 0
        for review in self:
            if review.correct_chatGPT(pattern):
                n += 1
        print("Num corrected:", n)

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

    def _split_by_label_has_usage_options(
        self,
        selection_strategy: ls.LabelSelectionStrategyInterface,
    ) -> tuple["ReviewSet", "ReviewSet"]:
        review_has_usage_options = lambda review: review.label_has_usage_options(
            selection_strategy
        )
        try:
            usage_option_reviews = self.filter(
                review_has_usage_options,
                inplace=False,
            )
            no_usage_option_reviews = self.filter(
                review_has_usage_options,
                inplace=False,
                invert=True,
            )
        except ValueError as e:
            raise ValueError(
                f"Not all reviews have labels for {selection_strategy}\n\t-> {e}"
            )
        return usage_option_reviews, no_usage_option_reviews

    def reduce_to_usage_option_split(
        self,
        selection_strategy: ls.LabelSelectionStrategyInterface,
        target_fraction: float,
        inplace: bool = False,
        seed: int = None,
    ) -> Optional["ReviewSet"]:
        def reduce_to_target_fraction(
            reducible_set: "ReviewSet",
            static_set: "ReviewSet",
            target_fraction: float,
            inplace: bool,
        ) -> Optional["ReviewSet"]:
            static_set_target_fraction = 1 - target_fraction
            target_size = len(static_set) / static_set_target_fraction

            reducible_set_target_size = round(target_size * target_fraction)
            assert reducible_set_target_size <= len(
                reducible_set
            ), "Reducible set needs to be larger than the target size (inferred from target fraction)"

            reducible_set = list(reducible_set)
            random.shuffle(reducible_set)

            reduced_reviews = ReviewSet.from_reviews(
                *reducible_set[:reducible_set_target_size], *static_set
            )
            if not inplace:
                return reduced_reviews
            # it is necessary to loop over all reviews in self to also remove those, that were discarded by _split_by_label_class
            for review in copy(self):
                if review not in reduced_reviews:
                    self.drop_review(review)

        if target_fraction < 0 or target_fraction > 1:
            raise ValueError("Target fraction must be between 0 and 1")

        if seed is not None:
            random.seed(seed)

        (
            usage_option_reviews,
            no_usage_option_reviews,
        ) = self._split_by_label_has_usage_options(selection_strategy)
        current_split = len(usage_option_reviews) / len(self)

        if current_split < target_fraction:
            return reduce_to_target_fraction(
                reducible_set=no_usage_option_reviews,
                static_set=usage_option_reviews,
                target_fraction=1 - target_fraction,
                inplace=inplace,
            )
        else:
            return reduce_to_target_fraction(
                reducible_set=usage_option_reviews,
                static_set=no_usage_option_reviews,
                target_fraction=target_fraction,
                inplace=inplace,
            )

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.get_data()["reviews"], orient="index")

    def get_dataloader(
        self,
        tokenizer,
        model_max_length: int,
        for_training: bool,
        selection_strategy: ls.LabelSelectionStrategyInterface = None,
        multiple_usage_options_strategy: Optional[str] = None,
        augmentation_set: Optional[str] = None,
        prompt_id: str = "avetis_v1",
        drop_out: float = 0.0,
        stratified_drop_out: bool = False,
        seed: int = None,
        rng: random.Random = random,
        **dataloader_args: dict,
    ) -> tuple[any, dict]:
        def are_all_valid_data_points(data_points: list[dict]) -> bool:
            for datapoint in data_points:
                # If None is in the values of a datapoint, the tokenized input or label was too long for the model
                # If selection_strategy is specified the output should not be 0 which is used as the default value
                if None in datapoint.values() or (
                    selection_strategy is not None and 0 in datapoint.values()
                ):
                    return False
            return True

        from torch.utils.data import DataLoader

        if seed is not None:
            rng.seed(seed)

        # Unpack all normal reviews and their augmentations into single ReviewSet
        all_reviews = (
            ReviewSet.from_reviews(
                *self,
                *itertools.chain(
                    *(
                        review.get_augmentations(selection_strategy, augmentation_set)
                        for review in self
                    )  # unpack interable with lists of augmented reviews
                ),
            )
            if augmentation_set is not None and selection_strategy is not None
            else copy(self)
        )
        for review in copy(all_reviews):
            review.tokenized_datapoints = list(
                review.get_tokenized_datapoints(
                    selection_strategy=selection_strategy,
                    tokenizer=tokenizer,
                    max_length=model_max_length,
                    for_training=for_training,
                    multiple_usage_options_strategy=multiple_usage_options_strategy,
                    prompt_id=prompt_id,
                )
            )
        valid_reviews = all_reviews.filter(
            lambda review: are_all_valid_data_points(review.tokenized_datapoints),
            inplace=False,
        )

        if stratified_drop_out and not selection_strategy:
            print(
                "Warning: Stratified drop out is only possible when a label is specified.\nResorting to simple random drop out."
            )
        dropped_reviews, remaining_reviews = (
            valid_reviews.stratified_split(drop_out, selection_strategy, rng=rng)
            if stratified_drop_out and selection_strategy
            else valid_reviews.split(drop_out, rng=rng)
        )

        tokenized_datapoints = [
            data_point
            for review in remaining_reviews
            for data_point in review.tokenized_datapoints
        ]
        rng.shuffle(tokenized_datapoints)
        return (
            DataLoader(tokenized_datapoints, **dataloader_args),
            {
                "selection_strategy": selection_strategy,
                "num_reviews": len(self),
                "num_augmented_reviews": len(all_reviews) - len(self),
                "num_invalid_reviews": len(all_reviews)
                - len(
                    valid_reviews
                ),  # invalid due to tokenized length or selection strategy
                "num_dropped_reviews": len(
                    dropped_reviews
                ),  # dropped due to random drop out
                "num_remaining_reviews": len(remaining_reviews),
                "num_datapoints": len(tokenized_datapoints),
            },
        )

    def split(
        self,
        fraction: float,
        seed: int = None,
        rng: random.Random = random,
    ) -> tuple["ReviewSet", "ReviewSet"]:
        if fraction < 0 or fraction > 1:
            raise ValueError("Fraction must be between 0 and 1")
        if fraction == 0:
            return ReviewSet.from_reviews(), ReviewSet.from_reviews(*self)
        if fraction == 1:
            return ReviewSet.from_reviews(*self), ReviewSet.from_reviews()

        if seed is not None:
            rng.seed(seed)

        reviews = sorted(list(self), key=lambda review: review.review_id)
        rng.shuffle(reviews)

        split_index = min(len(reviews) - 1, max(1, int(len(reviews) * fraction)))
        return (
            ReviewSet.from_reviews(*reviews[:split_index]),
            ReviewSet.from_reviews(*reviews[split_index:]),
        )

    def stratified_split(
        self,
        fraction: float,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        seed: Optional[int] = None,
        rng: random.Random = random,
    ) -> tuple["ReviewSet", "ReviewSet"]:
        """Split the ReviewSet while keeping ratio of label classes.

        Args:
            fraction (float): Fraction of reviews to be in the first set.
            label_selection_strategy (ls.LabelSelectionStrategyInterface): Label selection strategy that specifies which label to use for stratification.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        if seed is not None:
            rng.seed(seed)

        (
            usage_option_reviews,
            no_usage_option_reviews,
        ) = self._split_by_label_has_usage_options(label_selection_strategy)

        set1_usage, set2_usage = usage_option_reviews.split(fraction, rng=rng)
        set1_no_usage, set2_no_usage = no_usage_option_reviews.split(fraction, rng=rng)

        return set1_usage | set1_no_usage, set2_usage | set2_no_usage

    def create_dataset(
        self,
        dataset_name: str,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
    ) -> tuple["ReviewSet", dict]:
        reviews = deepcopy(
            self.filter_with_label_strategy(label_selection_strategy, inplace=False)
        )
        dataset_length = len(reviews)
        if dataset_length == 0:
            raise ValueError("There is no review that has any of the specified labels.")

        label_id_counts = {}
        for review in reviews:
            dataset_label_id = review.get_label_id_from_strategy(
                label_selection_strategy
            )
            dataset_label = review.get_label_for_id(dataset_label_id)

            for label_id, label in copy(review["labels"]).items():
                if label is not dataset_label:
                    del review["labels"][label_id]

            # when creating a dataset the datasets field in a review will only contain the dataset that is currently being created, same for test
            dataset_label["datasets"] = [dataset_name]
            if dataset_label_id not in label_id_counts:
                label_id_counts[dataset_label_id] = 1
            else:
                label_id_counts[dataset_label_id] += 1

        num_reviews_with_usage = len(
            reviews.filter(
                lambda review: review.label_has_usage_options(label_selection_strategy),
                inplace=False,
            )
        )

        return reviews, {
            "num_reviews": dataset_length,
            "num_reviews_for_labels": label_id_counts,
            "original_usage_split": round(num_reviews_with_usage / dataset_length, 3),
        }

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

    def remove_outliers(
        self,
        distance_threshold: float,
        remove_percentage: float,
        selection_strategy: DatasetSelectionStrategy,
    ):
        from clustering import utils
        from clustering.clusterer import Clusterer
        from clustering.data_loader import DataLoader
        import pandas as pd

        clustering_config = {
            "data": {
                "model_name": "all-mpnet-base-v2",
            },
            "clustering": {
                "use_reduced_embeddings": False,
                "algorithm": "agglomerative",
                "metric": "cosine",
                "linkage": "average",
                "save_to_disk": False,
                "distance_thresholds": [distance_threshold],
            },
        }

        if remove_percentage == 0:
            return

        arg_dicts = utils.get_arg_dicts(clustering_config, len(self))
        assert len(arg_dicts) == 1
        review_set_df, df_to_cluster = DataLoader(
            self, selection_strategy, clustering_config["data"]
        ).load()

        clustered_df = Clusterer(
            df_to_cluster, arg_dicts[0]
        ).cluster()  # arg_dicts[0] is the only arg_dict
        clustered_df = utils.merge_duplicated_usage_options(clustered_df, review_set_df)
        total_usage_options = len(clustered_df)

        count_label_df = (
            clustered_df.groupby("label")
            .count()
            .reset_index()
            .sort_values(ascending=True, by="review_id")
        )
        outlier_df = pd.DataFrame()
        for label in count_label_df["label"]:
            if outlier_df.shape[0] / total_usage_options >= remove_percentage:
                break
            outlier_df = outlier_df.append(clustered_df[clustered_df["label"] == label])
        review_ids_to_drop = set(outlier_df["review_id"].tolist())

        print(
            f"{outlier_df.shape[0]}/{total_usage_options} usage options are outliers -> removing {len(review_ids_to_drop)}/{len(self)} reviews"
        )
        for outlier_id in review_ids_to_drop:
            self.drop_review(outlier_id)

    def merge_labels(self, *label_ids: str, new_label_id: str) -> None:
        assert new_label_id not in self.get_all_label_ids()

        strategy = ls.LabelIDSelectionStrategy(*label_ids)
        for review in self:
            label = review.get_label_from_strategy(strategy)
            review.add_label(
                label_id=new_label_id,
                usage_options=label["usageOptions"],
                datasets=label["datasets"],
                metadata=label["metadata"],
            )

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        if path:
            self.save_path = path
        assert (
            self.save_path is not None
        ), "ReviewSet has no `save_path`; please supply a path when calling `save`"

        with open(self.save_path, "w") as file:
            json.dump(self.get_data(), file)
