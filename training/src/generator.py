import dataset
from torch.utils.data import DataLoader


class Generator:
    def __init__(self, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset.ReviewDataset(
            "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/golden_dataset_v3.json",
            tokenizer,
            max_length,
            evaluate=True,
        )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=32,
            num_workers=2,
        )

    def generate(self, model) -> None:
        for batch in self.data_loader:
            output = model.model.generate(
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
