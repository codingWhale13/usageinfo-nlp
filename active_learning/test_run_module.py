#!/usr/bin/env python3
"""
Use this file to run a integrated Active Learning Module without starting a real training run
"""
from lightning import pytorch as pl

from training.model import ReviewModel
from active_learning.module import ActiveLearningLossBasedSampler


from transformers import (
    T5Tokenizer,
)

a = ActiveLearningLossBasedSampler()
model = ReviewModel(
    model=None,
    model_name="Hello world",
    tokenizer=T5Tokenizer.from_pretrained("t5-small", model_max_length=512),
    max_length=512,
    hyperparameters={"batch_size": 8},
    data={"dataset_name": "bumsebiene-69", "validation_split": 0.5},
    multiple_usage_options_strategy="shuffle-3",
    seed=42,
    active_layers={"encoder": ":0", "decoder": ":0", "lm_head": False},
    optimizer=None,
    trainer=None,
    lr_scheduler_type=None,
    optimizer_args=None,
    gradual_unfreezing_mode=None,
    active_learning_module=a,
)
