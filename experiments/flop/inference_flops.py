#!/usr/bin/env python3

from training.generator import DEFAULT_GENERATION_CONFIG, Generator
from helpers.review_set import ReviewSet
import training.utils as utils

artifact_name = "llm-bayes-invsqrtlr-gpt4-0822154646"
generator = Generator(artifact_name, "greedy", checkpoint="best")
review_set = ReviewSet.from_files(
    utils.get_dataset_path("paper-gpt4-train"),
    utils.get_dataset_path("paper-gpt4-val"),
    "~/Projects/bsc2022-usageinfo/silver-v2.json",
)

num_tokens, flops = generator.generate_label(review_set, label_id="flop")

print(
    f"Number of tokens generated: {num_tokens}",
    f"Total number of FLOPs: {flops}",
    f"Average number of FLOPs per Token: {flops/num_tokens}",
    f"Total number of reqeusts: {len(review_set)}",
    f"Average number of FLOPs per request: {flops/len(review_set)}",
    sep="\n",
)
