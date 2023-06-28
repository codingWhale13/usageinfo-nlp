import json
import random

import yaml

EXPERIMENT_COUNT = 5
CONFIG_FOLDER = "/home/ubuntu/bsc2022-usageinfo/experiments/flan-t5-base"  # EC2 path

for experiment_id in range(1, EXPERIMENT_COUNT + 1):
    while True:
        active_encoder = random.randint(1, 8)
        active_decoder = random.randint(1, 8)
        if 4 <= active_encoder + active_decoder <= 12:
            break

    optimizer = random.choice(["AdamW", "AdaFactor"])
    lr_scheduler =random.choice([True, False])
    if lr_scheduler:
        lr_scheduler = "AdaFactor" if optimizer == "AdaFactor" else "OneCycleLR"

    config_str = f"""
accumulate_grad_batches: {random.randint(4, 16)}
active_layers:
  decoder: "-{active_encoder}:"
  encoder: "-{active_decoder}:"
  lm_head: "{random.choice([True, False])}"
active_learning: 
 module: "ActiveDataModule"
 parameters: 
module:
parameters: 
artifact:
  checkpoint: #9
  name:
batch_size: 16
cluster:
  devices: 1
  num_nodes: 1
dataset:
  test_set:
    name: "silver-test-69"
  training_set:
    augmentation_set:
    drop_out: 0.0
    name: "botched-6644"
    dataloader_setup_seed: 42
    stratified_drop_out: True
    usage_split:
    validation_split: 0.1
  validation_set:
    name:
epochs: 20
gradual_unfreezing_mode: ""
lr_scheduler_type: {lr_scheduler}
model_name: "flan-t5-base"
multiple_usage_options_strategy: "default"
optimizer:
  lr: {random.uniform(0.00005, 0.001)}
  name: {optimizer}
  relative_step: False
  scale_parameter: False
  warmup_init: null
  weight_decay: {random.uniform(0.0, 0.02)}
prompt_id: {random.choice(["avetis_v1", "original"])}
seed: 42
test_run: False
"""

    config = yaml.safe_load(config_str)
    with open(f"{CONFIG_FOLDER}/config_{experiment_id}.yml", "w") as file:
        config_yml = yaml.dump(config)
        file.write(config_yml)
