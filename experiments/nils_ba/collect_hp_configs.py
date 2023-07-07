import wandb
import yaml
import os

# run name ordered by config id (1, 2, ..., 35)
run_names = [
    "major-lake-738",
    "chocolate-morning-739",
    "fearless-violet-737",
    "magic-smoke-740",
    "radiant-sunset-722",
    "laced-puddle-732",
    "neat-yogurt-736",
    "peach-totem-730",
    "summer-glade-735",
    "blooming-spaceship-727",
    "misty-glade-767",
    "dandy-water-742",
    "clear-bee-743",
    "electric-butterfly-744",
    "summer-rain-745",
    "rare-frog-746",
    "valiant-haze-747",
    "eager-plant-775",
    "warm-rain-780",
    "quiet-violet-755",
    "scarlet-sea-782",
    "firm-wind-749",
    "happy-rain-783",
    "daily-hill-774",
    "zany-water-756",
    "gentle-brook-773",
    "different-snow-772",
    "trim-galaxy-758",
    "peachy-plant-762",
    "mild-universe-771",
    "misty-butterfly-766",
    "morning-armadillo-779",
    "light-valley-776",
    "olive-sea-777",
    "neat-dew-770",
]

api = wandb.Api()
runs = api.runs("bsc2022-usageinfo/rlp-t2t")
configs = {}
for run in runs:
    if run.name in run_names:
        if run.name in configs:
            print("ERROR: duplicate run name found!")
            exit()
        config_id = run_names.index(run.name) + 1
        configs[f"config-{config_id}_{run.name}"] = run.config

current_dir = os.path.dirname(os.path.join(os.path.realpath(__file__)))

hp_exp_id = 1
for config_name, config in sorted(
    configs.items(), key=lambda x: int(x[0].split("-")[-1])
):
    with open(
        os.path.join(current_dir, "experiment_hp", "configs3", f"{config_name}.yaml"),
        "w",
    ) as file:
        config_yml = yaml.dump(config)
        file.write(config_yml)
    hp_exp_id += 1
