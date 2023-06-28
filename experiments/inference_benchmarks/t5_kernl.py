from experiments.inference_benchmarks.model import AbstractBenchmarkModel
from kernl.model_optimization import optimize_model
import torch
from tqdm import tqdm


class OptimizedKernlModel(AbstractBenchmarkModel):
    def _initialize(self) -> float:
        optimize_model(self.model.encoder)
        optimize_model(self.model.decoder)

    def name(self):
        return "Kernl optimization"

    def _run(self, model_kwargs: dict) -> None:
        with torch.inference_mode(), torch.autocast(
            cache_enabled=True, device_type="cuda"
        ):
            for input_ids in tqdm(self.dataloader):
                input_ids = input_ids.to("cuda")
                self.model.generate(inputs=input_ids, **model_kwargs)
