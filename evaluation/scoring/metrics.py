from evaluation.scoring.core import get_most_similar, get_similarity
from evaluation.scoring.standard_metrics import bleu_score, sacrebleu_score, rouge_score
from evaluation.scoring import DEFAULT_METRICS
from evaluation.scoring.custom_metrics import (
    custom_f1_score,
    custom_recall,
    custom_precision,
    custom_f1_score_ak,
    custom_precision_ak,
    custom_recall_ak,
)
from statistics import mean
import math

DEFAULT_STRING_SIMILARITY = "all-mpnet-base-v2"
DEFAULT_AGG = mean
DEFAULT_NLP_THRESHOLD = 0.7


class NoUseCaseOptions:
    pass


class SingleReviewMetrics:
    def __init__(
        self,
        predictions: list,
        references: list,
        string_similarity=DEFAULT_STRING_SIMILARITY,
    ) -> None:
        self.predictions = predictions
        self.references = references
        self.string_similarity = string_similarity

    @classmethod
    def from_labels(
        cls,
        labels: dict[str, dict],
        prediction_label_id: str,
        reference_label_id: str,
        string_similarity: str = DEFAULT_STRING_SIMILARITY,
    ):
        return cls(
            predictions=labels[prediction_label_id]["usageOptions"],
            references=labels[reference_label_id]["usageOptions"],
            string_similarity=string_similarity,
        )

    def calculate(
        self,
        metric_ids=DEFAULT_METRICS,
    ) -> dict[str, float]:
        scores = {}

        for metric_id in metric_ids:
            try:
                metric_result = getattr(self, metric_id)()
            except ZeroDivisionError:
                metric_result = math.nan
            scores[metric_id] = metric_result

        return scores

    def custom_mean_recall(self):
        return custom_recall(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
            agg=mean,
        )

    def custom_mean_precision(self):
        return custom_precision(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
            agg=mean,
        )

    def custom_mean_f1(self):
        return custom_f1_score(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
            agg=mean,
        )

    def custom_min_precision(self):
        return custom_precision(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
            agg=min,
        )

    def custom_min_recall(self):
        return custom_recall(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
            agg=min,
        )

    def custom_min_f1(self):
        return custom_f1_score(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
            agg=min,
        )

    def custom_weighted_mean_f1(self):
        return custom_f1_score_ak(
            predictions=self.predictions,
            references=self.references,
            string_similarity=self.string_similarity,
        )

    def custom_weighted_mean_precision(self):
        return custom_precision_ak(
            self.predictions, self.references, self.string_similarity, mean
        )

    def custom_weighted_mean_recall(self):
        return custom_recall_ak(
            self.predictions, self.references, self.string_similarity, mean
        )

    def custom_classification_score(self):
        matches = {reference: [] for reference in self.references}
        non_matching_predictions = []

        for prediction in self.predictions:
            is_prediction_matched = False
            for reference in self.references:
                similarity = get_similarity(
                    prediction, reference, self.string_similarity
                )
                if similarity >= DEFAULT_NLP_THRESHOLD:
                    matches[reference].append(prediction)
                    is_prediction_matched = True
            if is_prediction_matched == False:
                non_matching_predictions.append(prediction)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TP"] = sum(
            [
                1 if len(matched_predictions) > 0 else 0
                for matched_predictions in matches.values()
            ]
        )
        results["FP"] = len(non_matching_predictions)
        results["TN"] = int(len(self.references) == len(self.predictions) == 0)
        results["FN"] = sum(
            [
                0 if len(matched_predictions) > 0 else 1
                for matched_predictions in matches.values()
            ]
        )

        return results

    def custom_classification_score_with_negative_class(self):
        matches = {reference: [] for reference in self.references}
        non_matching_predictions = []

        for prediction in self.predictions:
            is_prediction_matched = False
            for reference in self.references:
                similarity = get_similarity(
                    prediction, reference, self.string_similarity
                )
                if similarity >= DEFAULT_NLP_THRESHOLD:
                    print(similarity, prediction, reference)
                    matches[reference].append(prediction)
                    is_prediction_matched = True
            if is_prediction_matched == False:
                non_matching_predictions.append(prediction)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TN"] = math.inf
        if len(self.references) == 0:
            results["TP"] = int(len(self.references) == len(self.predictions) == 0)
            results["FP"] = len(self.predictions)
            results["FN"] = int(len(self.predictions) > 0)
            return results
        else:
            results["TP"] = sum(
                [
                    1 if len(matched_predictions) > 0 else 0
                    for matched_predictions in matches.values()
                ]
            )
            results["FP"] = len(non_matching_predictions)
            results["FN"] = sum(
                [
                    0 if len(matched_predictions) > 0 else 1
                    for matched_predictions in matches.values()
                ]
            )

            return results

    # not in use
    def custom_symmetric_similarity_classification_score_with_negative_class(self):
        best_matching_predictions = dict.fromkeys(self.references, 0)
        best_matching_references = dict.fromkeys(self.predictions, 0)

        for prediction in self.predictions:
            similarity, best_matched_reference = get_most_similar(
                prediction, self.references, self.string_similarity
            )
            best_matching_references[prediction] = (similarity, best_matched_reference)

        for reference in self.references:
            similarity, best_matching_prediction = get_most_similar(
                reference, self.predictions, self.string_similarity
            )
            best_matching_predictions[reference] = (
                similarity,
                best_matching_prediction,
            )

        if len(self.references) == 0:
            if len(self.predictions) == 0:
                best_matching_predictions[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_predictions[NoUseCaseOptions] = (0.0, None)

        if len(self.predictions) == 0:
            if len(self.references) == 0:
                best_matching_references[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_references[NoUseCaseOptions] = (0.0, None)

        return best_matching_predictions, best_matching_references

    def custom_similarity_classification_score_with_negative_class(self):
        best_matching_predictions = dict.fromkeys(self.references, 0)
        best_matching_references = dict.fromkeys(self.predictions, 0)

        for prediction in self.predictions:
            similarity, best_matched_reference = get_most_similar(
                prediction, self.references, self.string_similarity
            )
            best_matching_references[prediction] = (similarity, best_matched_reference)
        for reference in self.references:
            similarity, best_matching_prediction = get_most_similar(
                reference, self.predictions, self.string_similarity
            )
            best_matching_predictions[reference] = (
                similarity,
                best_matching_prediction,
            )

        if len(self.references) == 0:
            if len(self.predictions) == 0:
                best_matching_predictions[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_predictions[NoUseCaseOptions] = (0.0, None)

        if len(self.predictions) == 0:
            if len(self.references) == 0:
                best_matching_references[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_references[NoUseCaseOptions] = (0.0, None)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TN"] = math.inf

        results["TP"] = sum(
            [similarity for similarity, _ in best_matching_predictions.values()]
        )

        results["FN"] = sum(
            [1 - similarity for similarity, _ in best_matching_predictions.values()]
        )

        results["FP"] = sum(
            [1 - similarity for similarity, _ in best_matching_references.values()]
        )

        return results

    def bleu(self):
        return bleu_score(
            predictions=self.predictions,
            references=self.references,
        )

    def sacrebleu(self):
        return sacrebleu_score(
            predictions=self.predictions,
            references=self.references,
        )

    def rouge1(self):
        return rouge_score(
            predictions=self.predictions,
            references=self.references,
            rouge_score="rouge1",
        )

    def rouge2(self):
        return rouge_score(
            predictions=self.predictions,
            references=self.references,
            rouge_score="rouge2",
        )

    def rougeL(self):
        return rouge_score(
            predictions=self.predictions,
            references=self.references,
            rouge_score="rougeL",
        )

    def rougeLsum(self):
        return rouge_score(
            predictions=self.predictions,
            references=self.references,
            rouge_score="rougeLsum",
        )
