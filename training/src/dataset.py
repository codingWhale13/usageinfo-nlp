import os
import sys

from torch.utils.data import Dataset

path = os.path.dirname(os.path.realpath(__file__))
new_path_split = path.split(os.sep)[:-2] + ["utils"]
path = os.path.join(*new_path_split)
sys.path.append(path)
from extract_reviews import extract_reviews_with_usage_options_from_json


class ReviewDataset(Dataset):
    def __init__(self, file_location, tokenizer, max_length=float("inf")):
        self.file_location = file_location
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.data[index]

    def _build(self):
        def tokenize(text, is_input):
            tokens = self.tokenizer(text, return_tensors="pt", padding="max_length")
            if is_input:
                ids = tokens["input_ids"]
                # You need to set the pad tokens for the input to -100 for some Transformers (https://github.com/huggingface/transformers/issues/9770)>
                tokens["input_ids"] = ids[
                    ids[:, :] == self.tokenizer.pad_token_id
                ] = -100
            return tokens if len(tokens["input_ids"]) <= self.max_length else None

        df = extract_reviews_with_usage_options_from_json(self.file_location)
        df = df[["product_title", "review_body", "usage_options"]]
        df["input"] = df.apply(
            lambda x: tokenize(f"{x.product_title} || {x.review_body}", is_input=True),
            axis=1,
        )
        df["target"] = df.usage_options.apply(
            lambda x: tokenize(f"{' || '.join(x)}", is_input=False)
        )
        df.dropna(inplace=True)
        self.data = [tuple(row) for row in df[["input", "target"]].to_numpy()]
