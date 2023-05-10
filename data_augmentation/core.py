# %%
import abc
from typing import Union

import helpers.review as r
import helpers.label_selection as ls
from data_augmentation.batch import encode_batch, decode_batch, split_into_batches


# Augments a string or a list of strings, only overwrite augment_batch, metadata, max_batch_size in subclasses
class TextAugmentation(metaclass=abc.ABCMeta):
    def metadata(self) -> dict:
        return {}

    def max_batch_size(self) -> int:
        return 32

    def augment(
        self, texts: Union[str, list[str]]
    ) -> tuple[Union[str, list[str]], dict]:
        if isinstance(texts, list):
            if len(texts) > self.max_batch_size():
                results = []
                print(
                    f"Received data of length: {len(texts)}. Splitting into batches of {self.max_batch_size()}"
                )
                batches = split_into_batches(texts, self.max_batch_size())
                for i, batch in enumerate(batches):
                    results += self.augment_batch(batch)
                    print(f"Progress: {i+1} / {len(batches)}")
                return results, self.metadata()
            else:
                return self.augment_batch(texts), self.metadata()
        elif isinstance(texts, str):
            return self.augment_batch([texts])[0], self.metadata()
        else:
            raise ValueError("Value of texts is not str or list[str]")

    def augment_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError()


class TestTextAugmentation(TextAugmentation):
    def augment_batch(self, texts: list[str]) -> list[str]:
        return [f"Test: {text}" for text in texts]


class ReviewAugmentation(metaclass=abc.ABCMeta):
    def augment(
        self,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        *reviews: r.Review,
    ) -> None:
        raise NotImplementedError()


class PartialReviewAugementation(ReviewAugmentation):
    def __init__(
        self,
        text_augmentation: TextAugmentation,
        augmented_parts: list[str] = None,
    ) -> None:
        super().__init__()
        self.text_augmentation = text_augmentation
        self.augmented_parts = augmented_parts or ["review_body", "usageOptions"]

    def augment(
        self,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        *reviews: r.Review,
    ) -> None:
        data_to_augment = []
        for review in reviews:
            label = review.get_label_from_strategy(label_selection_strategy)
            augmentable_data = review.data | {"usageOptions": label["usageOptions"]}
            data_to_augment.append(
                {
                    part: augmentable_data[part]
                    for part in self.augmented_parts
                    if augmentable_data[part] != [] and augmentable_data[part] != ""
                }
            )
        instructions, batch = encode_batch(data_to_augment)
        augmented_batch, metadata = self.text_augmentation.augment(batch)
        augmented_data = decode_batch(augmented_batch, instructions)
        for review_text_augementations, review in zip(augmented_data, reviews):
            if len(review_text_augementations) > 0:
                label = review.get_label_from_strategy(label_selection_strategy)
                label["augmentations"] = label.get("augmentations", [])
                label["augmentations"].append(
                    review_text_augementations | {"metadata": metadata}
                )


class MultiAugmentation(ReviewAugmentation):
    def __init__(self, *review_augmentations: ReviewAugmentation) -> None:
        self.review_augmentations = review_augmentations

    def augment(
        self,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        *reviews: r.Review,
    ) -> None:
        for review_augmentation in self.review_augmentations:
            review_augmentation.augment(label_selection_strategy, *reviews)
