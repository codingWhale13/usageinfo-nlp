import itertools
import os
from statistics import mean

from evaluation.scoring.core import *


def custom_precision(
    predictions: list[str],
    references: list[str],
    string_similiarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        similarities = [
            get_most_similar(prediction, references, string_similiarity)[0]
            for prediction in predictions
        ]
        return agg(similarities)


def custom_recall(
    predictions: list[str],
    references: list[str],
    string_similiarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        similarities = [
            get_most_similar(reference, predictions, string_similiarity)[0]
            for reference in references
        ]
        return agg(similarities)


def custom_f1_score(
    predictions: list[str],
    references: list[str],
    string_similiarity: str = "all-mpnet-base-v2",
    agg: callable = mean,
) -> float:
    precision = custom_precision(predictions, references, string_similiarity, agg)
    recall = custom_recall(predictions, references, string_similiarity, agg)

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)
