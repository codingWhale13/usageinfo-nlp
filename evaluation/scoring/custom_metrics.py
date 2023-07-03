from statistics import mean

import numpy as np

from evaluation.scoring.core import *
from openai_api.openai_backend import DEFAULT_OPENAI_SIM_PARAMS

# NOTE: All metrics defined in this file have the same signature. This is required by `Metrics`.


def remove_duplicates(l: list, use_lowercase: bool = True) -> list:
    return list(set([item.lower() for item in l])) if use_lowercase else list(set(l))


def custom_precision(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    distance_metric: str = "cosine_relu",
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        predictions = remove_duplicates(predictions, use_lowercase)
        similarities = [
            get_most_similar(
                label=prediction,
                options=references,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                distance_metric=distance_metric,
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
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    distance_metric: str = "cosine_relu",
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        references = remove_duplicates(references, use_lowercase)
        similarities = [
            get_most_similar(
                label=reference,
                options=predictions,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                distance_metric=distance_metric,
            )[0]
            for reference in references
        ]
        return agg(similarities)


def custom_f1_score(
    predictions: list[str],
    references: list[str],
    comparator: str = "all-mpnet-base-v2",
    agg: callable = mean,
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    distance_metric: str = "cosine_relu",
) -> float:
    precision = custom_precision(
        predictions=predictions,
        references=references,
        comparator=comparator,
        agg=agg,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        distance_metric=distance_metric,
    )
    recall = custom_recall(
        predictions=predictions,
        references=references,
        comparator=comparator,
        agg=agg,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        distance_metric=distance_metric,
    )

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
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    distance_metric: str = "cosine_relu",
) -> float:
    if len(predictions) == 0:
        return int(len(references) == 0)
    else:
        predictions = remove_duplicates(predictions, use_lowercase)
        similarities = [
            get_most_similar(
                label=prediction,
                options=references,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                distance_metric=distance_metric,
            )[0]
            for prediction in predictions
        ]
        if len(predictions) == 1:
            weights = np.array(
                [1]
            )  # if there is only one prediction, we don't need to calculate weights
        else:
            similarity_matrix = [
                [
                    get_similarity(
                        label_1=prediction_1,
                        label_2=prediction_2,
                        comparator=comparator,
                        use_lowercase=use_lowercase,
                        openai_params=openai_params,
                        modification=modification,
                        distance_metric=distance_metric,
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
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    distance_metric: str = "cosine_relu",
) -> float:
    if len(references) == 0:
        return int(len(predictions) == 0)
    else:
        references = remove_duplicates(references, use_lowercase)
        similarities = [
            get_most_similar(
                reference,
                predictions,
                comparator=comparator,
                use_lowercase=use_lowercase,
                openai_params=openai_params,
                modification=modification,
                distance_metric=distance_metric,
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
                    get_similarity(
                        label_1=reference_1,
                        label_2=reference_2,
                        comparator=comparator,
                        use_lowercase=use_lowercase,
                        openai_params=openai_params,
                        modification=modification,
                        distance_metric=distance_metric,
                    )
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
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    distance_metric: str = "cosine_relu",
) -> float:
    precision = custom_precision_ak(
        predictions=predictions,
        references=references,
        comparator=comparator,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        distance_metric=distance_metric,
    )
    recall = custom_recall_ak(
        predictions=predictions,
        references=references,
        comparator=comparator,
        use_lowercase=use_lowercase,
        openai_params=openai_params,
        modification=modification,
        distance_metric=distance_metric,
    )

    if precision == recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)
