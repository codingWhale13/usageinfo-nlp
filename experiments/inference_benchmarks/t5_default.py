from experiments.inference_benchmarks.model import AbstractBenchmarkModel
import torch
from tqdm import tqdm


class TransformersDefault(AbstractBenchmarkModel):
    def name(self):
        return "T5 standard"

    def _run(self, model_kwargs: dict) -> None:
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            for input_ids in tqdm(self.dataloader):
                input_ids = input_ids.to("cuda")
                self.model.generate(inputs=input_ids, **model_kwargs)
