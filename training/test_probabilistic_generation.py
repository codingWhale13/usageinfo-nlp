# %%
from training.probablistic_generator import BatchProbabilisticGenerator
from training.generator import Generator
from helpers.review_set import ReviewSet
from scipy.stats import entropy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from helpers.label_selection import LabelIDSelectionStrategy
from active_learning.metrics.entropy import calculate_normalized_entropy

"""
generator = Generator("decent-elevator-481", "beam_search", 9, "all")
prob_generator = BatchProbabilisticGenerator(
    "decent-elevator-481", "beam_search", 9, "best"
)
"""
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

prob_generator = BatchProbabilisticGenerator(
    prompt_id="active_learning_v1", model=model, tokenizer=tokenizer
)
review_set_name = "silver-v1.json"
reviews = ReviewSet.from_files(review_set_name)
results = prob_generator.generate_usage_options_prob_based_batch(reviews)

selection_strategy = LabelIDSelectionStrategy("*")
df_data = []
for review_id, review in results.items():
    no_usage_options_prob = 0
    aggregated_results = {}

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
