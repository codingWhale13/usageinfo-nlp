import os
import sys
import argparse
import spacy
from statistics import mean


path = os.path.dirname(os.path.realpath(__file__))
new_path_split = path.split(os.sep)[:-1] + ["utils"]
sys.path.append(os.path.join(os.path.sep, *new_path_split))

from extract_reviews import extract_reviews_with_usage_options_from_json

DEFAULT_NLP_THRESHOLD = 0.7


def extract_review_with_id(df, review_id):
    review = df[df.review_id == review_id]
    if review.empty:
        return None
    return review.iloc[0]


def extract_labels(turker_df, golden_df):
    """Return for each review labelled by a turker a pair of two lists:
    turker labels and golden labels."""
    labels = []
    for _, turker_review in turker_df.iterrows():
        golden_review = extract_review_with_id(golden_df, turker_review.review_id)
        if golden_review is None:
            continue
        labels.append((turker_review.usage_options, golden_review.usage_options))

    return labels


def get_similarity(str_1, str_2, nlp):
    return nlp(str_1).similarity(nlp(str_2))


def get_most_similar(label, options, nlp, threshold_word_sim=0):
    """For a single `label`, find the most similar match from `options`.

    Returns tuple (best similarity score, option with best similiarity score)."""
    assert 0 <= threshold_word_sim <= 1

    result = (-1, None)
    for option in options:
        similarity = get_similarity(option, label, nlp)
        if similarity > result[0] and similarity >= threshold_word_sim:
            result = (similarity, option)

    return result


def get_similarity_for_review(
    primary_labels,
    secondary_labels,
    nlp,
    label_aggregation=mean,
    review_aggregation=mean,
    threshold_word_sim=0,
    threshold_label_sim=0,
):
    """Calculate for one review how close the primary labels are to the secondary labels."""
    aggregations = [min, max, mean]
    if label_aggregation not in aggregations or review_aggregation not in aggregations:
        raise ValueError("Invalid aggregation function")

    similarity_per_secondary_label = {label: [] for label in secondary_labels}
    for primary_label in primary_labels:
        similarity, secondary_label = get_most_similar(
            primary_label, secondary_labels, nlp, threshold_word_sim
        )
        if similarity >= threshold_label_sim and secondary_label is not None:
            similarity_per_secondary_label[secondary_label].append(similarity)

    similarity_per_secondary_label = {
        usage: label_aggregation(values) if len(values) > 0 else 0
        for usage, values in similarity_per_secondary_label.items()
    }

    return review_aggregation(similarity_per_secondary_label.values())


def get_similarities_for_review_batch(
    labels,
    nlp,
    label_aggregation=mean,
    review_aggregation=mean,
    threshold_word_sim=0,
    threshold_label_sim=0,
    threshold_review_sim=0,
    turker_first=True,
):
    """Calculate the similarity between each pair of turker labels and golden labels.
    label_aggregation is the function used to aggregate the similarity scores for each label
    review_aggregation is the function used to aggregate the similarity scores for each review
    threshold_word_sim is the minimum similarity between two words to consider them similar
    threshold_label_sim is the minimum similarity between two labels to consider them similar
    threshold_review_sim is the minimum similarity of a review to consider it similar
    If turker_first is True, the first set of reviews is the set of reviews labelled by the turker
    """
    assert (
        0 <= threshold_label_sim <= 1
        and 0 <= threshold_review_sim <= 1
        and 0 <= threshold_word_sim <= 1
    )

    scores = []  # will be filled with 0 or 1 values

    for turker_labels, golden_labels in labels:
        if len(turker_labels) == 0 or len(golden_labels) == 0:
            scores.append(int(len(turker_labels) == len(golden_labels)))
        else:
            primary_labels, secondary_labels = turker_labels, golden_labels
            if not turker_first:
                primary_labels, secondary_labels = secondary_labels, primary_labels

            similarity = get_similarity_for_review(
                primary_labels=primary_labels,
                secondary_labels=secondary_labels,
                nlp=nlp,
                label_aggregation=label_aggregation,
                review_aggregation=review_aggregation,
                threshold_word_sim=threshold_word_sim,
                threshold_label_sim=threshold_label_sim,
            )

            if threshold_review_sim == 0:
                scores.append(similarity)
            else:
                scores.append(int(similarity >= threshold_review_sim))

    return scores


def turker_score(labels, nlp, threshold=DEFAULT_NLP_THRESHOLD):
    """Determine quality of labels by averaging over pass (1) or fail (0) for each review

    `threshold` is the minimum mean label similarity required for a review to be passed
    """
    return mean(
        get_similarities_for_review_batch(
            labels, nlp, threshold_review_sim=threshold, turker_first=True
        )
    )


def precision(labels, nlp, threshold=DEFAULT_NLP_THRESHOLD):
    """Answers the question: How many turker labels are golden?

    `threshold` represents minimum similarity between labels to consider them
    close enough"""
    assert 0 <= threshold <= 1

    retrieved, relevant = 0, 0

    for turker_labels, golden_labels in labels:
        retrieved += len(turker_labels)
        for turker_label in turker_labels:
            similarity, _ = get_most_similar(turker_label, golden_labels, nlp)
            if similarity >= threshold:
                relevant += 1

    return relevant / retrieved


def recall(labels, nlp, threshold=DEFAULT_NLP_THRESHOLD):
    """Answers the question: How many golden labels have been identified?

    `threshold` represents minimum similarity between labels to consider them
    close enough"""
    assert 0 <= threshold <= 1

    retrieved, relevant = 0, 0

    for turker_labels, golden_labels in labels:
        relevant += len(golden_labels)
        for golden_label in golden_labels:
            similarity, _ = get_most_similar(golden_label, turker_labels, nlp)
            if similarity >= threshold:
                retrieved += 1

    return retrieved / relevant


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        "-l",
        required=True,
        help="JSON file containing manually labelled reviews",
    )
    parser.add_argument(
        "--golden",
        "-g",
        required=True,
        help="JSON file containing golden review labels",
    )

    args = parser.parse_args()

    golden_df = extract_reviews_with_usage_options_from_json(args.golden)

    turker_df = extract_reviews_with_usage_options_from_json(args.labels)
    labels = extract_labels(turker_df, golden_df)
    nlp = spacy.load("en_core_web_md")

    print(f"Turker Score: {turker_score(labels, nlp)}")
