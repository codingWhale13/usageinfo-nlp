import abc
import fnmatch
from typing import Optional

from review import Review


class LabelSelectionStrategyInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "retreive_label") and callable(subclass.retreive_label)

    def retreive_label(self, review: Review) -> Optional[dict]:
        raise NotImplementedError


class LabelIDSelectionStrategy(LabelSelectionStrategyInterface):
    def __init__(self, *label_ids: str):
        self.label_ids = label_ids

    def retreive_label(self, review: Review) -> Optional[dict]:
        for label_id in self.label_ids:
            matches = fnmatch.filter(review.get_label_ids(), label_id)
            if matches:
                return review.get_label(matches[0])

        return None
