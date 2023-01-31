from statistics import mean

"""
INSTALL spacy model
python -m spacy download en_core_web_md
"""

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
    return nlp(str_1.lower()).similarity(nlp(str_2.lower()))


def get_most_similar(label, options, nlp, threshold_word_sim=0) -> tuple[float, str]:
    """For a single `label`, find the most similar match from `options`.

    Returns tuple (best similarity score, option with best similiarity score)."""
    assert 0 <= threshold_word_sim <= 1

    result = (0, None)
    for option in options:
        similarity = get_similarity(option, label, nlp)
        if similarity >= max(result[0], threshold_word_sim):
            result = (similarity, option)

    return result


def custom_precision(labels, nlp, agg: callable = min) -> list[float]:
    result = []

    for turker_labels, golden_labels in labels:
        if len(turker_labels) == 0:
            result.append(1)
        else:
            similarities = []
            for turker_label in turker_labels:
                similarity, _ = get_most_similar(turker_label, golden_labels, nlp)
                similarities.append(similarity)
            result.append(agg(similarities))

    return result


def custom_recall(labels: list[tuple], nlp, agg: callable = min) -> list[float]:
    result = []

    for turker_labels, golden_labels in labels:
        if len(golden_labels) == 0:
            result.append(1)
        else:
            similarities = []
            for golden_label in golden_labels:
                similarity, _ = get_most_similar(golden_label, turker_labels, nlp)
                similarities.append(similarity)
            result.append(agg(similarities))

    return result
