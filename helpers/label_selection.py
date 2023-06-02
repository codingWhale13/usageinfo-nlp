from __future__ import annotations
import abc
import fnmatch
from typing import Optional, Union
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helpers.review import Review


class LabelSelectionStrategyInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "retrieve_label") and callable(subclass.retrieve_label)

    def retrieve_label(self, review: Review) -> Optional[dict]:
        raise NotImplementedError

    def retrieve_label_id(self, review: Review) -> Optional[str]:
        raise NotImplementedError


class LabelIDSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *label_ids: str):
        self.label_ids = label_ids

    def retrieve_label(self, review: Review) -> Optional[dict]:
        label_id = self.retrieve_label_id(review)
        if label_id is not None:
            return review.get_labels()[label_id]
        return None

    def retrieve_label_id(self, review: Review) -> Optional[str]:
        review_label_ids = review.get_label_ids()
        for label_id in self.label_ids:
            matches = fnmatch.filter(review_label_ids, label_id)
            if matches:
                return matches[0]
        return None


class MultiLabelIDSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *label_ids: str):
        self.label_ids = label_ids

    def retrieve_label(self, review: Review) -> Optional[list[dict]]:
        label_ids = self.retrieve_label_id(review)
        if label_ids is not None:
            labels = review.get_labels()
            return [labels[label_id] for label_id in label_ids]
        return None

    def retrieve_label_id(self, review: Review) -> Optional[list[str]]:
        review_label_ids = review.get_label_ids()
        return list(set(review_label_ids).intersection(set(self.label_ids)))


class DatasetSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *datasets: Union[str, tuple[str, str]]):
        self.datasets = datasets

    def retrieve_label(self, review) -> Optional[dict]:
        label_id = self.retrieve_label_id(review)
        if label_id is not None:
            return review.get_labels()[label_id]
        return None

    def retrieve_label_id(self, review: Review) -> Optional[str]:
        for dataset in self.datasets:
            dataset_name, dataset_part = (
                dataset if isinstance(dataset, tuple) else (dataset, None)
            )

            for label_id, label in review.get_labels().items():
                datasets = label["datasets"]
                if dataset_name in datasets:
                    if not dataset_part or dataset_part == datasets[dataset_name]:
                        return label_id

        return None
