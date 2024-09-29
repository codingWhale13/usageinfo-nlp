#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean

from src.evaluation.plotting.plot_custom_metrics import plot_f1


def bar_plot(score_values, save_path, relevant_label_ids="[ALL]"):
    for score_id, scores in score_values.items():
        mean_values = []
        for label_id, values in scores.items():
            if relevant_label_ids == "[ALL]" or label_id in relevant_label_ids:
                mean_values.append((mean(values), label_id))
        mean_values.sort()
        x = [i[1] for i in mean_values]
        y = [i[0] for i in mean_values]
        fig = plt.figure(figsize=(10, 12))
        plt.title(score_id)
        plt.xticks(rotation=45, ha="right")
        sns.barplot(x=x, y=y)
        plt.savefig(Path(save_path, f"{score_id}_barplot.png"))


path = "..."
save_path = Path().absolute()

ref_id = "golden_v2"

score_data = {}
with open(path) as file:
    data = json.load(file)

    for review in data["reviews"]:
        for label_id in review["labels"]:
            if label_id == ref_id:
                continue
            if label_id not in score_data:
                score_data[label_id] = {}

            metadata = review["labels"][label_id]["metadata"]
            if "scores" in metadata:
                for key, value in metadata["scores"][ref_id].items():
                    if key not in score_data[label_id]:
                        score_data[label_id][key] = []
                    score_data[label_id][key].append(value)

# print scores per label_id
for label_id, scores in score_data.items():
    print(f"\n{label_id}:")
    for score_id, values in scores.items():
        print(f"{score_id}: {mean(values)} (averaged over {len(values)} values)")

# reorganize scores by score id
score_values = {}
for label_id, scores in score_data.items():
    for score_id, values in scores.items():
        if score_id not in score_values:
            score_values[score_id] = {}
        score_values[score_id][label_id] = values

for score_id, scores in score_values.items():
    for score_id, scores in score_values.items():
        for label_id, values in scores.items():
            plot_f1(scores[label_id], score_id, save_path=save_path, base_name=label_id)

bar_plot(score_values, save_path)
