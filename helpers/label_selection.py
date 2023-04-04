import abc
import fnmatch
from typing import Optional, Union


class LabelSelectionStrategyInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "retrieve_label") and callable(subclass.retrieve_label)

    def retrieve_label(self, review) -> Optional[dict]:
        raise NotImplementedError


class LabelIDSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *label_ids: str):
        self.label_ids = label_ids

    def retrieve_label(self, review) -> Optional[dict]:
        review_label_ids = review.get_label_ids()
        for label_id in self.label_ids:
            matches = fnmatch.filter(review_label_ids, label_id)
            if matches:
                return review.get_labels()[matches[0]]

        return None


class DatasetSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *datasets: Union[str, tuple[str, str]]):
        self.datasets = datasets

    def retrieve_label(self, review) -> Optional[dict]:
        for dataset in self.datasets:
            dataset_name, dataset_part = (
                dataset if isinstance(dataset, tuple) else (dataset, None)
            )

            for label in review.get_labels().values():
                datasets = label["datasets"]
                if dataset_name in datasets:
                    if not dataset_part or dataset_part == datasets[dataset_name]:
                        return label

        return None
