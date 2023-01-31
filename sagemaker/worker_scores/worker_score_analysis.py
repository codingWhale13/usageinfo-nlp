# %%
import matplotlib.pyplot as plt
from worker_scorer import calculate_golden_dataset_scores

imerit = [
    "../bsc2022-usageinfo-imerit-vendor-test-run-output.json",
    "../bsc2022-usageinfo-imerit-vendor-test-run-2-clone.json",
]
cogito = [
    "../bsc2022-usageinfo-vendor-test-run-output.json",
    "../bsc2022-usageinfo-cogito-vendor-test-run-2.json",
]
golden_dataset = "../golden_dataset/v3.json"


scores_imerit = []
scores_cogito = []
for i in range(min(len(imerit), len(cogito))):
    scores_imerit.append(calculate_golden_dataset_scores(imerit[i], golden_dataset))
    scores_cogito.append(calculate_golden_dataset_scores(cogito[i], golden_dataset))


# %%
runs = list(range(1, len(scores_imerit) + 1))
runs, scores_cogito

#%%
metrics = ["miss_rate", "balanced_accuracy"]

fig, ax = plt.subplots()
for metric in metrics:
    plt.plot(
        runs,
        [x["total"][metric] for x in scores_cogito],
        "o",
        label=f"Cogito: {metric}",
        linestyle=":",
    )
    plt.plot(
        runs,
        [x["total"][metric] for x in scores_imerit],
        "o",
        label=f"IMeritf: {metric}",
        linestyle=":",
    )
plt.legend()
plt.show()
