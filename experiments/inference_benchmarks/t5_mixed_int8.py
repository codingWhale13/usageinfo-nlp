from experiments.inference_benchmarks.model import AbstractBenchmarkModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm


class MixedInt8(AbstractBenchmarkModel):
    def __init__(self, n_samples: int, batch_size: int = 1024):
        super().__init__(n_samples, batch_size)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME, load_in_8bit=True
        ).to("cuda")

    def name(self):
        return "mixed_int8"

    def _run(self, model_kwargs: dict) -> None:
        with torch.inference_mode():
            for input_ids in tqdm(self.dataloader):
                input_ids = input_ids.to("cuda")
                self.model.generate(inputs=input_ids, **model_kwargs)
