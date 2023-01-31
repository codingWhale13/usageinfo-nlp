import dataset
import torch
from torch.utils.data import DataLoader

import utils


class Generator:
    def __init__(self, artifact_name, dataset_version: str, checkpoint=None) -> None:
        checkpoint = torch.load(
            utils.get_model_path({"name": artifact_name, "checkpoint": checkpoint})
        )
        (
            self.model,
            self.tokenizer,
            self.max_length,
        ) = utils.get_model_config_from_checkpoint(checkpoint["model"], checkpoint)
        self.dataset = dataset.ReviewDataset(
            utils.get_dataset_paths(dataset_version)["test_dataset"],
            self.tokenizer,
            self.max_length,
            evaluate=True,
        )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=8,
            num_workers=2,
        )

    def generate(self) -> None:
        for batch in self.data_loader:
            output = self.model.generate(
                input_ids=batch[0]["input_ids"],
                attention_mask=batch[0]["attention_mask"],
            )
            predictions = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            texts = self.tokenizer.batch_decode(
                batch[0]["input_ids"], skip_special_tokens=True
            )
            labels = self.tokenizer.batch_decode(
                batch[1]["input_ids"], skip_special_tokens=True
            )
            for i in range(len(predictions)):
                print(f"Text: {texts[i]}")
                print(f"Prediction: {predictions[i]}")
                print(f"Label: {labels[i]}\n")
