from statistics import mean

import numpy as np

from evaluation.scoring.core import *
from openai_api.openai_backend import DEFAULT_OPENAI_SIM_PARAMS

# NOTE: All metrics defined in this file have the same signature. This is required by `Metrics`.


def custom_precision(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        similarities = [
            get_most_similar(
                prediction, references, comparator, use_lowercase, openai_params
            )[0]
            for prediction in predictions
        ]
        return agg(similarities)


def custom_recall(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        similarities = [
            get_most_similar(
                reference,
                predictions,
                comparator,
                use_lowercase,
                openai_params=openai_params,
            )[0]
            for reference in references
        ]
        return agg(similarities)


def custom_f1_score(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
) -> float:
    precision = custom_precision(
        predictions, references, comparator, agg, openai_params
    )
    recall = custom_recall(predictions, references, comparator, agg, openai_params)

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


# the following "ak" scores were designed by Avetis and Konstantin


def custom_precision_ak(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        predictions = list(
            set([prediction.lower() for prediction in predictions])
        )  # remove duplicates
        similarities = [
            get_most_similar(
                prediction,
                references,
                comparator,
                use_lowercase,
                openai_params=openai_params,
            )[0]
            for prediction in predictions
        ]
        if len(predictions) == 1:
            weights = np.array(
                [1]
            )  # if there is only one reference, we don't need to calculate weights
        else:
            similarity_matrix = [
                [
                    get_similarity(
                        prediction_1, prediction_2, comparator, openai_params
                    )
                    for prediction_2 in predictions
                ]
                for prediction_1 in predictions
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
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        references = list(
            set([reference.lower() for reference in references])
        )  # remove duplicates
        similarities = [
            get_most_similar(
                reference,
                predictions,
                comparator,
                use_lowercase,
                openai_params=openai_params,
            )[0]
            for reference in references
        ]
        if len(references) == 1:
            weights = np.array(
                [1]
            )  # if there is only one reference, we don't need to calculate weights
        else:
            similarity_matrix = [
                [
                    get_similarity(reference_1, reference_2, comparator, openai_params)
                    for reference_2 in references
                ]
                for reference_1 in references
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
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
) -> float:
    precision = custom_precision_ak(
        predictions, references, comparator, use_lowercase, openai_params
    )
    recall = custom_recall_ak(
        predictions, references, comparator, use_lowercase, openai_params
    )

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)
