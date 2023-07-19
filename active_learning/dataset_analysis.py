from helpers.review_set import ReviewSet
from helpers.review import Review
from helpers.label_selection import AbstractLabelSelectionStrategy
from statistics import mean
import pandas as pd

def word_count(string: str) -> int:
    return len(string.split(" "))

def analyze_review_set(review_set: ReviewSet, label_selection_strategy: AbstractLabelSelectionStrategy) -> pd.DataFrame:
    data = []
    for review in review_set:
        data.append(analyze_review(review, label_selection_strategy))
    return pd.DataFrame.from_records(data)

def analyze_review(
    review: Review, label_selection_strategy: AbstractLabelSelectionStrategy
):
    usageOptions = review.get_label_from_strategy(label_selection_strategy)[
        "usageOptions"
    ]
    usageOptionsWordCount = [0] + [
        word_count(usageOption) for usageOption in usageOptions
    ]

    return {
        "review_id": review.review_id,
        "review_body": review.data["review_body"],
        "review_headline": review.data["review_headline"],
        "product_title": review.data["product_title"],
        "product_category": review.data["product_category"],
        "star_rating": int(review.data["star_rating"]),
        "usage_options_count": len(usageOptions),
        "has_usage_options": len(usageOptions) > 0,
        "input_word_count": word_count(review.get_prompt()),
        "usage_options_word_count": sum(usageOptionsWordCount),
        "average_usage_option_word_count": mean(usageOptionsWordCount),
        "usage_options": usageOptions,
    }
