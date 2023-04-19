import itertools
import os
from statistics import mean

import numpy as np

from evaluation.scoring.core import *


def custom_precision(
    predictions: list[str],
    references: list[str],
    string_similarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        similarities = [
            get_most_similar(prediction, references, string_similarity)[0]
            for prediction in predictions
        ]
        return agg(similarities)


def custom_recall(
    predictions: list[str],
    references: list[str],
    string_similarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        similarities = [
            get_most_similar(reference, predictions, string_similarity)[0]
            for reference in references
        ]
        return agg(similarities)


def custom_f1_score(
    predictions: list[str],
    references: list[str],
    string_similarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    precision = custom_precision(predictions, references, string_similarity, agg)
    recall = custom_recall(predictions, references, string_similarity, agg)

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


# NEW SCORES FROM AVETIS AND KONSTI:


def custom_precision_ak(
    predictions: list[str],
    references: list[str],
    string_similarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        predictions = list(
            set([prediction.lower() for prediction in predictions])
        )  # remove duplicates
        similarities = [
            get_most_similar(prediction, references, string_similarity)[0]
            for prediction in predictions
        ]
        if len(predictions) == 1:
            weights = np.array(
                [1]
            )  # if there is only one reference, we don't need to calculate weights
        else:
            similarity_matrix = [
                [get_similarity(prediction, prediction2) for prediction2 in predictions]
                for prediction in predictions
            ]
            weights = np.array(
                [
                    (1 - (sum(similarity_matrix[i])) / (len(similarity_matrix[i])))
                    for i in range(len(similarity_matrix))
                ]
            )

        return np.dot(similarities, weights) / sum(weights)


def custom_recall_ak(
    predictions: list[str],
    references: list[str],
    string_similarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        references = list(
            set([reference.lower() for reference in references])
        )  # remove duplicates
        similarities = [
            get_most_similar(reference, predictions, string_similarity)[0]
            for reference in references
        ]
        if len(references) == 1:
            weights = np.array(
                [1]
            )  # if there is only one reference, we don't need to calculate weights
        else:
            similarity_matrix = [
                [get_similarity(reference, reference2) for reference2 in references]
                for reference in references
            ]
            weights = np.array(
                [
                    (1 - (sum(similarity_matrix[i])) / (len(similarity_matrix[i])))
                    for i in range(len(similarity_matrix))
                ]
            )

        return np.dot(similarities, weights) / sum(weights)


def custom_f1_score_ak(
    predictions: list[str],
    references: list[str],
    string_similarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    precision = custom_precision_ak(predictions, references, string_similarity, agg)
    recall = custom_recall_ak(predictions, references, string_similarity, agg)

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)
