import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Iterable, Union
from pathlib import Path

from src.review_set import ReviewSet
import src.helpers.label_selection as ls


WORD_COUNT_CATEGORIES = {
    "very short": [0, 15],
    "short": [16, 25],
    "medium": [26, 40],
    "long": [41, 90],
    "very long": [91, 100000],
}
METRIC_ID = "custom_weighted_mean_f1"
TP_SAMPLE_SIZE = 10
FP_SAMPLE_SIZE = 10
HARD_SAMPLE_SIZE = 10
FN_SAMPLE_SIZE = 10

HARD_REVIEWS = [
    "R2ZOYVY2FQW5G",
    "R3CQGAL7I5VFXV",
    "RZWYL8GEI1HP5",
    "R1VNIJS0NPRSX7",
    "RLXN3DIK61JMP",
    "REP5SXKAHMV78",
    "R2FA4LIRAML82P",
]
ORDERS = {
    "star_rating": [1, 2, 3, 4, 5],
    "review_length": list(WORD_COUNT_CATEGORIES.keys()),
}


def get_scored_reviews_dataframe(
    label_id: Union[str, ls.LabelSelectionStrategyInterface],
    review_set: ReviewSet,
    *reference_label_ids: Union[str, ls.LabelSelectionStrategyInterface],
) -> pd.DataFrame:
    reviews_list = []
    for review in review_set:
        review_scores = review.get_scores(
            label_id, *reference_label_ids, metric_ids=[METRIC_ID]
        )
        if not review_scores:
            continue

        reference_labels = []
        no_usage_options_ok, usage_options_ok = False, False
        for reference_label_id in reference_label_ids:
            if issubclass(type(reference_label_id), ls.AbstractLabelSelectionStrategy):
                reference_label = review.get_label_from_strategy(reference_label_id)
            else:
                reference_label = review.get_label_for_id(reference_label_id)
            if reference_label:
                reference_labels.append(reference_label["usageOptions"])
                if len(reference_label["usageOptions"]) > 0:
                    usage_options_ok = True
                else:
                    no_usage_options_ok = True

        if issubclass(type(label_id), ls.AbstractLabelSelectionStrategy):
            label_id = label_id.retrieve_label_id(review)
        prediction_has_usage_options = (
            len(review.get_label_for_id(label_id)["usageOptions"]) > 0
        )

        if prediction_has_usage_options:
            usage_class = "TP" if usage_options_ok else "FP"
        else:
            usage_class = "TN" if no_usage_options_ok else "FN"

        reviews_list.append(
            {
                "review_id": review.review_id,
                "review": f'Product title: {review.data["product_title"]}\nHeadline: {review.data["review_headline"]}\n{review.data["review_body"]}',
                "review_body": review.data["review_body"],
                "usage_class": usage_class,
                "predicted_usage_options": "; ".join(
                    review.get_label_for_id(label_id)["usageOptions"]
                ),
                "reference_usage_options": "\n".join(
                    [
                        "; ".join(reference_labels[i])
                        for i in range(len(reference_labels))
                    ]
                ),
                "star_rating": review.data["star_rating"],
                "product_category": review.data["product_category"],
                METRIC_ID: review_scores[METRIC_ID],
                "hard_review": review.review_id in HARD_REVIEWS,
            }
        )

    df = pd.DataFrame.from_records(reviews_list)
    # Correct star rating because some are saved as int vs some as str
    df["star_rating"] = df["star_rating"].apply(lambda x: int(x))
    return df


def get_score_report(
    review_set: ReviewSet,
    folder: str,
    label_id: Union[str, ls.LabelSelectionStrategyInterface],
    *reference_label_ids: Union[str, ls.LabelSelectionStrategyInterface],
):
    print(
        f"Generating report for {label_id} against {', '.join(reference_label_ids)} and save it in folder: {folder}"
    )
    Path(folder).mkdir(parents=True, exist_ok=True)

    reviews_df = get_scored_reviews_dataframe(
        label_id, review_set, *reference_label_ids
    )

    # Preprocessing for plotting
    aggregations = ["count", "mean", "std"]

    total_scores_dict = reviews_df.agg({METRIC_ID: aggregations}).to_dict()[METRIC_ID]

    usage_options_count_df = (
        reviews_df.groupby("usage_class").size().reset_index(name="count")
    )

    tp_score = (
        reviews_df[reviews_df["usage_class"] == "TP"]
        .agg({METRIC_ID: aggregations})
        .to_dict()[METRIC_ID]
    )

    reviews_df["word_count"] = reviews_df["review_body"].apply(lambda x: len(x.split()))
    reviews_df["review_length"] = reviews_df["word_count"].apply(
        lambda x: next(
            (
                category
                for category, interval in WORD_COUNT_CATEGORIES.items()
                if interval[0] <= x <= interval[1]
            )
        )
    )

    groups = {}

    groups["review_length"] = get_data_per_category(
        reviews_df,
        "review_length",
        aggregations,
        explanation=f"Reviews are divided in these categories based on their length in words: {WORD_COUNT_CATEGORIES}",
    )

    groups["star_rating"] = get_data_per_category(
        reviews_df,
        "star_rating",
        aggregations,
    )

    groups["product_category"] = get_data_per_category(
        reviews_df,
        "product_category",
        aggregations,
    )

    reviews_sample = {
        "FP": get_reviews(reviews_df, "usage_class", "FP", FP_SAMPLE_SIZE),
        "FN": get_reviews(reviews_df, "usage_class", "FN", FN_SAMPLE_SIZE),
        "TP": get_reviews(reviews_df, "usage_class", "TP", TP_SAMPLE_SIZE),
        "HARD": get_reviews(reviews_df, "hard_review", True, HARD_SAMPLE_SIZE),
    }

    reviews_df["reference_usage_options"] = reviews_df["reference_usage_options"].apply(
        lambda x: x.replace("\n", " | ")
    )
    reviews_df.to_csv(f"{folder}/reviews.csv")
    return (
        label_id,
        reference_label_ids,
        METRIC_ID,
        total_scores_dict,
        usage_options_count_df,
        tp_score,
        groups,
        folder,
        reviews_sample,
    )


def get_data_per_category(df, category, aggregations, explanation=""):
    """Returns a dictionary with plots, tables and explanation for a given category"""
    df = df.sort_values(by=[category])

    score_info = (
        df[df["usage_class"] == "TP"].groupby([category]).agg({METRIC_ID: aggregations})
    )
    count_info = df.groupby([category, "usage_class"]).size().reset_index(name="count")

    return {
        "plots": [
            get_score_plot(df, category),
            get_count_plot(count_info, category),
        ],
        "tables": [score_info, count_info],
        "explanation": explanation,
    }


def sort_df(df, category: str, reset_index=False):
    def _sort(sort_by):
        if "usage_class" in df.columns:
            df.sort_values(
                by=[sort_by, "usage_class"], inplace=True, ascending=[True, False]
            )
        else:
            df.sort_values(by=[sort_by], inplace=True, ascending=True)

    if reset_index:
        df.reset_index(inplace=True)

    if category in ORDERS:
        df["order"] = df[category].apply(lambda x: ORDERS[category].index(x))
        _sort("order")
        df.drop(["order"], axis=1, inplace=True)
    else:
        _sort(category)


def get_score_plot(df, category: str):
    sort_df(df, category)

    plt.figure()
    plt.clf()
    plot = sns.barplot(
        df[df["usage_class"] == "TP"],
        x=category,
        y=METRIC_ID,
        errorbar="se",
    )
    # This is needed since the labels are too long to fit horizontally side by side
    if category == "product_category":
        plt.xticks(rotation=90)

    plot.set(title=f'F1 score for "true positives" grouped by {category}')

    return plot


def get_count_plot(df, category: str):
    sort_df(df, category)

    plt.figure()
    plt.clf()
    plot = sns.barplot(
        df,
        x=category,
        y="count",
        hue="usage_class",
        errorbar="se",
        palette=["#90EE90", "#013220", "#ffcccb", "#8b0000"],  # green to red
    )
    if category == "product_category":
        plt.xticks(rotation=90)

    plot.set(title=f"review count grouped by {category}")

    return plot


def get_reviews(df, category: str, class_label: str, n: Optional[int] = None):
    df = df[df[category] == class_label]
    if n:
        n = min(n, len(df))
        df = df.sample(n=n)
    return df[
        [
            "review_id",
            "review",
            "predicted_usage_options",
            "reference_usage_options",
            "star_rating",
            "product_category",
            METRIC_ID,
        ]
    ]
