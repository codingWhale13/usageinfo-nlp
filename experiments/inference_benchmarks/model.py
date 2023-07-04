import abc
import time
import statistics
from typing import Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class AbstractBenchmarkModel(abc.ABC):
    MODEL_NAME = "google/flan-t5-base"

    def __init__(self, n_samples: int, batch_size: int = 512):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def _initialize(self) -> float:
        """Perform any additional initialization/ optimizations"""
        pass

    def bechmark_initialization(self) -> tuple[float, float]:
        start_time = time.perf_counter()
        self._initialize()
        return time.perf_counter() - start_time, 0.0

    def name(self) -> str:
        raise NotImplementedError

    def benchmark_tokenize(self, input: list[str]) -> tuple[float, float]:
        print("Benchmarking tokenization")
        execution_times = []
        for _ in range(self.n_samples):
            start_time = time.perf_counter()
            self._tokenize(input)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times)

        return mean_time, std_time

    def _tokenize(self, input: list[str]) -> None:
        input_ids = self.tokenizer(
            input,
            return_tensors="pt",
            pad_to_multiple_of=8,
            padding=True,
            truncation=True,
        )["input_ids"]
        self.dataloader = DataLoader([x for x in input_ids], batch_size=self.batch_size)

    def _run(self, model_kwargs: dict) -> None:
        """
        Run the model on the given input.

        Args:
            model_input: The input for the model.
        """
        raise NotImplementedError

    def benchmark(self, model_kwargs: dict) -> Tuple[float, float]:
        """
        Run the model multiple times and measure the execution time.

        Args:
            n_samples: The number of times to run the model.
            model_input: The input for the model.

        Returns:
            A tuple containing the mean and standard deviation of the execution time in seconds.
        """
        execution_times = []
        print("Warming up model")
        for _ in range(1):
            self._run(model_kwargs)
        print("Benchmarking model runtime")
        for _ in range(self.n_samples):
            start_time = time.perf_counter()
            self._run(model_kwargs)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times)

        return mean_time, std_time
