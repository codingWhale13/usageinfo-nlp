from helpers.review_set import ReviewSet
from transformers.modeling_outputs import Seq2SeqLMOutput
from training.model import ReviewModel

from random import shuffle
from torch.utils.data import DataLoader

NUM_WORKERS = 4


class EncoderModel(ReviewModel):
    def __init__(self, *args, **kwargs):
        super(EncoderModel, self).__init__(*args, **kwargs)

    def _step(self, batch) -> Seq2SeqLMOutput:
        labels = batch["output"]["input_ids"]
        outputs = self.encoder(
            input_ids=batch["input"]["input_ids"],
            attention_mask=batch["input"]["attention_mask"],
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        self.active_data_module.process_step(batch_idx, batch, outputs)
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        return outputs.loss

    def train_dataloader(self):
        rs = ReviewSet.from_files("/home/codingwhale/Documents/Studium/HPI/Materialien/BP/bsc2022-usageinfo/data/ba/50k_sample_encoded.json")

        tokenized_datapoints = []
        for i, review in enumerate(rs):
            tokens = self.tokenizer(
                f"Product title: {review['product_title']} \nReview body: {review['review_body']}\n",
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            tokenized_datapoints.append(tokens)


        shuffle(tokenized_datapoints)

        general_dataloader_args = {
            "batch_size": self.hyperparameters["batch_size"],
            "num_workers": NUM_WORKERS,
            "multiple_usage_options_strategy": self.multiple_usage_options_strategy,
            "prompt_id": self.prompt_id,
        }

        return DataLoader(tokenized_datapoints, **general_dataloader_args)