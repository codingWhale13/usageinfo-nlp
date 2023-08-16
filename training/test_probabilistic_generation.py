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
)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

"""
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")
tokenizer = PreTrainedTokenizerFast.from_pretrained("google/flan-t5-base")


generator = Generator(
    "car-barfs-stupid-155-7", "greedy", "best", prompt_id="active_learning_v1"
)
"""
score_data = []
data = []
data2 = []
review_set_name = "ba-30k-train-reviews.json"  # "golden_small.json"  # hard_reviews.json"  # "silver-v1.json"
reviews, _ = ReviewSet.from_files(review_set_name).split(0.05)
selection_strategy = DatasetSelectionStrategy("ba-30k-train")
for i in range(0, 32):
    print("Loading model:", f"greedy_entropy_run_1-{i}-")
    prob_generator = BatchProbabilisticGenerator(
        prompt_id="active_learning_v1",
        artifact_name=f"greedy_entropy_b16_run_2-{i}_",
        checkpoint="best",
        batch_size=64,
        token_top_k=10,
        minimum_probability=0.01,
        max_iterations=100,
        max_sequence_length=32,
    )
    """
    prob_generator = BatchProbabilisticGenerator(
        prompt_id="active_learning_v1",
        model=model,
        tokenizer=tokenizer,
        batch_size=512,
        token_top_k=5,
    )
    """

    results = prob_generator.generate_usage_options_prob_based_batch(reviews)
    for review_id, review in results.items():
        probs = [x["probability"] for x in review]
        data.append(
            {
                "review_id": review_id,
                "entropy_lowest_probability": calculate_lowest_probability_approximation_entropy(
                    probs
                ),
                "entropy_normalized": calculate_normalized_entropy(probs),
                "has_usage_options": reviews[review_id].label_has_usage_options(
                    selection_strategy
                ),
                "iteration": i,
            }
        )
        data2.append(
            {
                "review_id": review_id,
                "entropy": calculate_lowest_probability_approximation_entropy(probs),
                "entropy_calculation_method": "maximum_predicted",
                "has_usage_options": reviews[review_id].label_has_usage_options(
                    selection_strategy
                ),
                "iteration": i,
            }
        )
        data2.append(
            {
                "review_id": review_id,
                "entropy": calculate_normalized_entropy(probs),
                "entropy_calculation_method": "normalized",
                "has_usage_options": reviews[review_id].label_has_usage_options(
                    selection_strategy
                ),
                "iteration": i,
            }
        )

    df = pd.DataFrame.from_records(data2)
    plt.clf()
    g = sns.FacetGrid(df, row="iteration", col="entropy_calculation_method")
    g.map_dataframe(sns.histplot, x="entropy", hue="has_usage_options")
    plt.savefig(f"entropy_report_2.png")

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
