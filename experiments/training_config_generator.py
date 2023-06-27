import json
import random

import yaml

while True:
    active_encoder = random.randint(1, 8)
    active_decoder = random.randint(1, 8)
    if 4 <= active_encoder + active_decoder <= 12:
        break

optimizer = random.choice(["AdamW", "AdaFactor"])
if random.choice([True, False]):
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
artifact:
  checkpoint: #9
  name: #"scintillating-mandu-99"
batch_size: 16
cluster:
  devices: 1 #Number of gpus per node
  num_nodes: 1
dataset:
  test_set:
    name: "silver-test-69"
  training_set:
    augmentation_set:
    drop_out: 0.0 # fraction of data to drop from training set (validation set stays untouched)
    name: "botched-6644"
    dataloader_setup_seed: 42
    stratified_drop_out: True
    usage_split: # 0.5
    validation_split: 0.0
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
print(json.dumps(config, indent=4))
