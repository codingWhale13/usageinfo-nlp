from helpers.review import Review
from helpers.label_selection import LabelSelectionStrategyInterface
from statistics import mean


def word_count(text: str):
    return len(text.split(" "))


def vectorize(
    review: Review, label_selection_strategy: None = LabelSelectionStrategyInterface
):
    label = review.get_label_from_strategy(label_selection_strategy)
    return [
        review.data["product_category"],
        int(review.data["star_rating"]),
        word_count(review.data["review_body"]),
        len(label["usageOptions"]),
        0
        if len(label["usageOptions"]) == 0
        else mean(map(lambda x: word_count(x), label["usageOptions"])),
        label["scores"][""],
    ]


def columns():
    return [
        "product_category",
        "star_rating",
        "review_body_word_count",
        "usage_options_count",
        "mean_usage_option_word_count",
        "f1_score",
    ]
