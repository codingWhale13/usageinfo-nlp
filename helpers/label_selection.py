from __future__ import annotations
import abc
import fnmatch
from typing import Optional, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helpers.review import Review


class LabelSelectionStrategyInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "retrieve_label") and callable(subclass.retrieve_label)

    def retrieve_label(self, review: Review) -> Optional[dict]:
        label_ids = self.retrieve_label_ids(review)
        if len(label_ids) > 0:
            labels = review.get_labels()
            return labels[label_ids[0]]
        return None

    def retrieve_label_id(self, review: Review) -> Optional[str]:
        label_ids = self.retrieve_label_ids(review)
        if len(label_ids) > 0:
            return label_ids[0]
        return None

    def retrieve_labels(self, review: Review) -> list[dict]:
        label_ids = self.retrieve_label_ids(review)
        labels = review.get_labels()
        return [labels[label_id] for label_id in label_ids]

    def retrieve_label_ids(self, review: Review) -> list[str]:
        raise NotImplementedError


class LabelIDSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *label_ids: str):
        self.label_ids = label_ids

    def retrieve_label_ids(self, review: Review) -> list[str]:
        review_label_ids = review.get_label_ids()
        label_ids = []
        for label_id in self.label_ids:
            matches = fnmatch.filter(review_label_ids, label_id)
            if matches:
                label_ids += matches
        return label_ids


class DatasetSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *datasets: Union[str, tuple[str, str]]):
        self.datasets = datasets

    def retrieve_label_ids(self, review: Review) -> list[str]:
        label_ids = []
        for dataset in self.datasets:
            dataset_name, dataset_part = (
                dataset if isinstance(dataset, tuple) else (dataset, None)
            )

            for label_id, label in review.get_labels().items():
                datasets = label["datasets"]
                if dataset_name in datasets:
                    if not dataset_part or dataset_part == datasets[dataset_name]:
                        label_ids.append(label_id)

        return label_ids
