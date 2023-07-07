import csv
import os
import yaml

all_values = []
FOLDER_PATH = "/experiments/nils_ba/experiment_hp"

config_id = 0
directory = os.fsencode(os.path.join(FOLDER_PATH, "configs"))
for yaml_file in sorted(
    os.listdir(directory), key=lambda x: int(str(x).split("_")[0].split("-")[1])
):
    config_id += 1
    filename = os.fsdecode(yaml_file)

    assert filename.endswith(".yaml") and filename.startswith(f"config-{config_id}")

    with open(os.path.join(FOLDER_PATH, "configs", filename), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    values = {}
    values["accumulated gradient batches"] = config["accumulate_grad_batches"]
    values[
        "unfrozen layers (encoder/decoder)"
    ] = f'{config["active_layers"]["decoder"][1:-1 ]}/{config["active_layers"]["encoder"][   1:-1 ]}'  # remove - and :
    values["use LM head"] = config["active_layers"]["lm_head"]
    values["learning rate optimizer"] = config["optimizer"]["name"]
    values["learning rate"] = config["optimizer"]["lr"]
    values["weight decay"] = config["optimizer"]["weight_decay"]
    values["learning rate scheduler"] = config["lr_scheduler_type"]
    values["Prompt ID"] = config["prompt_id"]

    all_values.append(values)


with open(os.path.join(FOLDER_PATH, "all_configs.csv"), "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(all_values[0].keys())
    for row in all_values:
        writer.writerow(row.values())
