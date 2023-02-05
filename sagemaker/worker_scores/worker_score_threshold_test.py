# %%
import os 
import sys
path = os.path.dirname(os.path.realpath(__file__))
new_path_split = path.split(os.sep)[:-2]
sys.path.append(os.path.join(os.path.sep, *new_path_split))

from utils.extract_reviews import extract_reviews_with_usage_options_from_json



# %%
from worker_metrics import Metrics
labels_file =  "../job_outputs/pre_labelled_bsc2022-usageinfo-openai.json"
golden_dataset = "../golden_datasets/v3.json"

labels = extract_reviews_with_usage_options_from_json(
        labels_file, use_predicted_usage_options=True
    )

scores = Metrics(labels, golden_dataset, threshold=0.7).calculate(['recall', 'precision'])

scores

# %%
import numpy as np
thresholds = list(np.linspace(0.5, 1.0, num=40, endpoint=False))

models = ["sentence-transformers/sentence-t5-base", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
threshold_scores = {}
for model in models:
    threshold_scores[model] = []
    for threshold in thresholds:
        score = Metrics(labels, golden_dataset, threshold=threshold, model_checkpoint=model).calculate(['recall', 'precision'])
        threshold_scores[model].append(score)
        print(score)
#%%
import matplotlib.pyplot as plt

metrics = ['recall']
fig, ax = plt.subplots()

for model in threshold_scores.keys():
    for metric in metrics:
        plt.plot(
            thresholds,
            [x[metric] for x in threshold_scores[model]],
            "o",
            label=f"{model}: {metric}",
            linestyle=":",
        )
plt.legend()
plt.show()
