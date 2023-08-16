# %%
from training.probablistic_generator import (
    BatchProbabilisticGenerator,
)
from training.generator import Generator
from helpers.review_set import ReviewSet
from scipy.stats import entropy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast
from helpers.label_selection import LabelIDSelectionStrategy, DatasetSelectionStrategy
from active_learning.metrics.entropy import (
    calculate_normalized_entropy,
    calculate_lowest_probability_approximation_entropy,
    aggregate_cluster_probabilities,
    calculate_lowest_probability_approximation_predictor_entropy
)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json


review_set_name = "ba-30k-train-reviews.json"  # "golden_small.json"  # hard_reviews.json"  # "silver-v1.json"
reviews = ReviewSet.from_files(review_set_name).filter(lambda x: x.review_id in ["R1HUV2P9DILZIF","R2R54RJFG1CAD7",
"R27EMRRS7E5EIC",
"R1HKFTB5KYKTXR",
"R19BVNDQ62M4AO"], inplace=False)
selection_strategy = DatasetSelectionStrategy("ba-30k-train")
data = {}
for review_id, _ in reviews.items():
    data[review_id] = []

import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def format_sequence(review_sequences):
    for x in review_sequences:
        del x["decoder_token_ids"]
    return review_sequences

from scipy.stats import entropy
def kl(review_sequences, review_seqeunces_2):
    probs = {}
    for i, review in enumerate([review_sequences, review_seqeunces_2]):
        for x in review:
            key = tuple(x["decoder_token_ids"])
            if key in probs:
                probs[key][i] = x["probability"]
            else:
                base_prob = 1*10**-10
                probs[key] = [base_prob, base_prob]
                probs[key][i] = x["probability"]
    probs_1 = [x[0] for x in probs.values()] 
    probs_2 = [x[1] for x in probs.values()]
    return entropy(probs_1, probs_2)

for i in [5, 6, 7, 8]:
    print("Loading model:", f"greedy_entropy_run_1-{i}-")
    prob_generator = BatchProbabilisticGenerator(
        prompt_id="active_learning_v1",
        artifact_name=f"greedy_entropy_b32_run_1-{i}_",
        checkpoint="best",
        batch_size=64,
        token_top_k=10,
        minimum_probability=0.01,
        max_iterations=100,
        max_sequence_length=32,
    )

    results = prob_generator.generate_usage_options_prob_based_batch(reviews)
    for review_id, review in results.items():
        probs = [x["probability"] for x in review]
        clustered_probs = aggregate_cluster_probabilities(review)
        main_part, predictor = calculate_lowest_probability_approximation_predictor_entropy(probs)
        clustered_main_part, clustered_predictor = calculate_lowest_probability_approximation_predictor_entropy(clustered_probs)

        data[review_id].append(
            {
                "sequences": sorted(format_sequence(review), key = lambda x: x["probability"], reverse=True),
                "maximum_predicted_entropy": {"value": calculate_lowest_probability_approximation_entropy(
                    probs
                ),
                "main_part": main_part,
                "predictor": predictor},
                "entropy_normalized": calculate_normalized_entropy(probs),
                "least_confidence": 1 - max(probs),
                "clustered_maximum_predicted_entropy": {"value": calculate_lowest_probability_approximation_entropy(clustered_probs),
                    "main_part": clustered_main_part,
                    "predictor": clustered_predictor
                },
                "cluster_normalized_entropy": calculate_normalized_entropy(clustered_probs),
                "labels": reviews[review_id].get_label_from_strategy(
                    selection_strategy
                )["usageOptions"],
                "iteration": i,
            }
        )
        """
        length = len(data[review_id])
        if length > 1:
            data[review_id][length -1]["kl"] = kl(data[review_id][length -1]["sequences"], data[review_id][length -2]["sequences"])
            del data[review_id][length -2]["sequences"]
        """
with open("predictions_sample_training_dataset_8_new.json", "w") as f:
    json.dump(data, f, cls=NpEncoder)
        
exit()
df = pd.DataFrame.from_records(score_data)
print(df)
df.to_csv("greedy_scores_new.csv")
sns.lineplot(df, x="iteration", y="mean")
plt.errorbar(
    x="iteration",
    y="mean",
    yerr="variance",
    data=df,
    fmt="none",
    ecolor="red",
    capsize=4,
)
plt.savefig("greedy_scores_new.png", dpi=300)
exit()
results = prob_generator.generate_usage_options_prob_based_batch(
    reviews,
    cluster_results=True,
)
for review_id, review in results.items():
    sorted_review = sorted(
        review, key=lambda r: (r["cluster"], r["probability"]), reverse=True
    )

    probs = {}
    for x in review:
        try:
            probs[x["cluster"]] += x["probability"]
        except KeyError:
            probs[x["cluster"]] = x["probability"]

    print(
        review_id,
        calculate_lowest_probability_approximation_entropy(list(probs.values())),
    )

    for usage_option in sorted_review:
        print(
            usage_option["probability"],
            usage_option["cluster"],
            usage_option["usageOptions"],
        )
selection_strategy = LabelIDSelectionStrategy("*")
df_data = []
exit()
# generator.generate_label(reviews, verbose=True)
for review_id, review in results.items():
    no_usage_options_prob = 0
    aggregated_results = {}

    print("prob", review_id)
    for x in review:
        if "usageOptions" in x:
            print(x["usageOptions"])
        else:
            print(x["probability"], x["decoder_token_ids"])

    continue
    for x in review:
        if tuple(set(x["usageOptions"])) not in aggregated_results:
            aggregated_results[tuple(set(x["usageOptions"]))] = x["probability"]
        else:
            aggregated_results[tuple(set(x["usageOptions"]))] += x["probability"]
    aggregated_results = [
        {"usageOptions": list(key), "probability": value}
        for key, value in aggregated_results.items()
    ]
    probs = [x["probability"] for x in aggregated_results]
    # probs.append(1 - sum(probs))

    sorted_usage_options = sorted(
        aggregated_results, key=lambda x: x["probability"], reverse=True
    )

    lc = 1 - sorted_usage_options[0]["probability"]
    review_entropy = calculate_normalized_entropy(probs)
    print(review_id, review_entropy, lc)
    for x in sorted_usage_options:
        print(x["probability"], x["usageOptions"])
    df_data.append(
        {
            "entropy": review_entropy,
            "least_confidence": lc,
            "review_id": review_id,
            "has_usage_options": len(
                reviews.get_review(review_id).get_label_from_strategy(
                    selection_strategy
                )["usageOptions"]
            )
            > 0,
        }
    )


exit()
import pandas as pd
import seaborn as sns

df = pd.DataFrame.from_records(df_data)
print(df)
print("Mean entropy: ", df["entropy"].mean())
print("Mean usage option entropy:", df[df["has_usage_options"]]["entropy"].mean())
print("no usage option mean", df[~df["has_usage_options"]]["entropy"].mean())


print("Mean lc:", df["least_confidence"].mean())
print("Usage options:", df[df["has_usage_options"]]["least_confidence"].mean())
print("No usage options:", df[~df["has_usage_options"]]["least_confidence"].mean())
sns.histplot(df, x="entropy").get_figure().savefig(f"{review_set_name}-entropy.png")
