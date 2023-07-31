# %%
from training.probablistic_generator_dynamic_length import (
    DynamicLengthBatchProbabilisticGenerator,
)
from training.probablistic_generator import (
    BatchProbabilisticGenerator,
)
from training.generator import Generator
from helpers.review_set import ReviewSet
from scipy.stats import entropy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast
from helpers.label_selection import LabelIDSelectionStrategy
from active_learning.metrics.entropy import (
    calculate_normalized_entropy,
    calculate_lowest_probability_approximation_entropy,
)


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")
tokenizer = PreTrainedTokenizerFast.from_pretrained("google/flan-t5-base")


generator = Generator(
    "car-barfs-stupid-155-7", "greedy", "best", prompt_id="active_learning_v1"
)


prob_generator = BatchProbabilisticGenerator(
    prompt_id="active_learning_v1",
    artifact_name="car-barfs-stupid-155-7",
    checkpoint="best",
    batch_size=512,
    token_top_k=5,
    minimum_probability=0.01,
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

review_set_name = "hard_reviews.json"  # hard_reviews.json"  # "silver-v1.json"
reviews = ReviewSet.from_files(review_set_name)


results = prob_generator.generate_usage_options_prob_based_batch(
    reviews,
    decode_results=True,
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
