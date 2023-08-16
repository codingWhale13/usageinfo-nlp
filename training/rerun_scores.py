# %%
from generator import DEFAULT_GENERATION_CONFIG, Generator
from helpers.review_set import ReviewSet
from helpers.label_selection import LabelIDSelectionStrategy
from active_learning.metrics.entropy import (
    calculate_normalized_entropy,
    calculate_lowest_probability_approximation_entropy,
)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from training.utils import get_model_dir_file_path

"""
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")
tokenizer = PreTrainedTokenizerFast.from_pretrained("google/flan-t5-base")


generator = Generator(
    "car-barfs-stupid-155-7", "greedy", "best", prompt_id="active_learning_v1"
)
"""
score_data = []

review_set = ReviewSet.from_files("silver_v1_newly_labeled.json")
selection_strategy = LabelIDSelectionStrategy("bp-silver*")


def unpack_string_dict(str_dict: dict) -> dict:
    result = {}
    for key, value in str_dict.items():
        if type(value) == dict:
            for inner_key, inner_value in value.items():
                result[key + "_" + inner_key] = inner_value
        else:
            result[key] = value
    return result


# print("random_baseline_b32_run_1-0" in review_set.get_all_label_ids())  #
# greedy_entropy_b32__random_baseline_b32_run_1_random_baseline_b32_run_2_random_baseline_b32_run_3_greedy_clustered_normalized_entropy_b32_run_1_subset_clustered_entropy_b32_run_4
for artifact_name in [
    "greedy_entropy_b32_run_1",
    "greedy_clustered_entropy_b32_run_1",
    "random_baseline_b32_run_1",
    "random_baseline_b32_run_2",
    "random_baseline_b32_run_3",
]:  # "greedy_clustered_normalized_entropy_b32_run_1", "subset_clustered_entropy_b32_run_4"]:
    run_scores = []
    base_dir = f"{artifact_name}-active_learning_dir"

    existing_scores = pd.read_csv(get_model_dir_file_path(base_dir, "scores.csv"))
    max_iterations = existing_scores["active_learning_iteration"].max()
    # print(max_iterations)
    #existing_scores.to_csv(get_model_dir_file_path(base_dir, "scores_backup.csv"))
    for iteration in range(1, max_iterations + 1):
        artifact_iteration_name = f"{artifact_name}-{iteration - 1}"
        print(artifact_iteration_name)
        if artifact_iteration_name not in review_set.get_all_label_ids():
            raise Exception(f"Label not found: {artifact_iteration_name}")
            #review_set.save("silver_v1_newly_labeled.json")
        scores = review_set.get_agg_scores(
            LabelIDSelectionStrategy(artifact_iteration_name), selection_strategy
        )

        new_scores = unpack_string_dict(scores) | {
            "active_learning_iteration": iteration
        }
        run_scores.append(new_scores)

    existing_scores.update(pd.DataFrame.from_records(run_scores), overwrite=True)
    existing_scores.to_csv(get_model_dir_file_path(base_dir, "scores_new_3.csv"))

exit()
# for x in ["entropy_lowest_probability", "entropy_normalized"]:
#    plt.clf()
#    sns.histplot(df, x=x, hue="has_usage_options")
#    plt.savefig(f"entropy_{x}_{i}.png")
# reviews.save()
# reviews.save()

""""
scores = reviews.get_agg_scores(label_id, DatasetSelectionStrategy("ba-30k-test"))
print(i, scores)
score_data.append(
    {
        "mean": scores["custom_weighted_mean_f1"]["mean"],
        "variance": scores["custom_weighted_mean_f1"]["variance"],
        "iteration": i,
    }
)
"""

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
