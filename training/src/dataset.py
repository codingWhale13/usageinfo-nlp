import torch
from typing import Tuple

from torch.utils.data import Dataset
import pandas as pd

import utils
from helpers.review_set import ReviewSet


class ReviewDataset(Dataset):
    def __init__(
        self, data: pd.DataFrame, tokenizer, max_length=float("inf"), evaluate=False
    ):
        self.dataframe = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.evaluate = evaluate

        self._build()

    @classmethod
    def from_dataset_name(cls, dataset_name, *args, **kwargs):
        dataset_location = utils.get_dataset_path(dataset_name)
        train_data, test_data = ReviewSet.from_files(dataset_location).get_dataset(
            dataset_name=dataset_name
        )

        train_data_df = pd.DataFrame.from_dict(train_data, orient="index")
        test_data_df = pd.DataFrame.from_dict(test_data, orient="index")

        return cls(data=train_data_df, *args, **kwargs), cls(
            data=test_data_df, *args, **kwargs
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _build(self):
        def tokenize(text, is_input):
            tokens = self.tokenizer(text, return_tensors="pt", padding="max_length")
            # Remove batch dimension, since we only have one example
            tokens["input_ids"] = tokens["input_ids"][0]
            tokens["attention_mask"] = tokens["attention_mask"][0]
            if not is_input and not self.evaluate:
                ids = tokens["input_ids"]
                # You need to set the pad tokens for the input to -100 for some Transformers (https://github.com/huggingface/transformers/issues/9770)>
                tokens["input_ids"][ids[:] == self.tokenizer.pad_token_id] = -100
            return tokens if len(tokens["input_ids"]) <= self.max_length else None

        df = self.dataframe[["product_title", "review_body", "usage_options"]]
        df["input"] = df.apply(
            lambda x: tokenize(
                f"Product title: {x.product_title} \n Review body: {x.review_body}",
                is_input=True,
            ),
            axis=1,
        )
        df["target"] = df.usage_options.apply(
            lambda x: tokenize(f"{', '.join(x)}", is_input=False)
        )
        df.dropna(inplace=True)
        self.data = [tuple(row) for row in df[["input", "target"]].to_numpy()]

    def split(self, fraction: float) -> Tuple[Dataset, Dataset]:
        assert 0 < fraction < 1
        dataset1_size = int(len(self) * fraction)
        dataset2_size = len(self) - dataset1_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self,
            [dataset1_size, dataset2_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, val_dataset
