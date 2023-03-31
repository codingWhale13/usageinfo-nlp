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
        cls, labels: dict[str, dict], prediction_label_id: str, reference_label_id: str
    ):
        return cls(
            labels[prediction_label_id]["usageOptions"],
            labels[reference_label_id]["usageOptions"],
        )

    def calculate(
        self,
        metric_ids=DEFAULT_METRICS,
    ) -> dict[str, float]:
        scores = {}

        for metric_id in metric_ids:
            try:
                metric_result = getattr(self, metric_id)(
                    self.predictions, self.references
                )
            except ZeroDivisionError:
                metric_result = math.nan
            scores[metric_id] = metric_result

        return scores

    def custom_mean_recall(self, pred, ref):
        return custom_recall(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
            agg=mean,
        )

    def custom_mean_precision(self, pred, ref):
        return custom_precision(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
            agg=mean,
        )

    def custom_mean_f1(self, pred, ref):
        return custom_f1_score(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
            agg=mean,
        )

    def custom_min_precision(self, pred, ref):
        return custom_precision(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
            agg=min,
        )

    def custom_min_recall(self, pred, ref):
        return custom_recall(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
            agg=min,
        )

    def custom_min_f1(self, pred, ref):
        return custom_f1_score(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
            agg=min,
        )

    def custom_weighted_mean_f1(self, pred, ref):
        return custom_f1_score_ak(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similarity,
        )

    def custom_weighted_mean_precision(self, pred, ref):
        return custom_precision_ak(pred, ref, self.string_similarity, mean)

    def custom_weighted_mean_recall(self, pred, ref):
        return custom_recall_ak(pred, ref, self.string_similarity, mean)

    def custom_classification_score(
        self, predictions: list[str], references: list[str]
    ):
        THRESHOLD = 0.8

        matches = {reference: [] for reference in references}
        non_matching_predictions = []

        for prediction in predictions:
            is_prediction_matched = False
            for reference in references:
                similarity = get_similarity(
                    prediction, reference, self.string_similarity
                )
                if similarity >= THRESHOLD:
                    matches[reference].append(prediction)
                    is_prediction_matched = True
            if is_prediction_matched == False:
                non_matching_predictions.append(prediction)

        print(matches, non_matching_predictions)
        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TP"] = sum(
            [
                1 if len(matched_predictions) > 0 else 0
                for matched_predictions in matches.values()
            ]
        )
        results["FP"] = len(non_matching_predictions)
        results["TN"] = int(len(references) == len(predictions) == 0)
        results["FN"] = sum(
            [
                0 if len(matched_predictions) > 0 else 1
                for matched_predictions in matches.values()
            ]
        )

        return results

    def custom_classification_score_with_negative_class(
        self, predictions: list[str], references: list[str]
    ):
        THRESHOLD = 0.85

        matches = {reference: [] for reference in references}
        non_matching_predictions = []

        print("Starting")
        for prediction in predictions:
            is_prediction_matched = False
            for reference in references:
                similarity = get_similarity(
                    prediction, reference, self.string_similarity
                )
                if similarity >= THRESHOLD:
                    print(similarity, prediction, reference)
                    matches[reference].append(prediction)
                    is_prediction_matched = True
            if is_prediction_matched == False:
                non_matching_predictions.append(prediction)

        print(matches, non_matching_predictions)
        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TN"] = math.inf
        if len(references) == 0:
            results["TP"] = int(len(references) == len(predictions) == 0)
            results["FP"] = len(predictions)
            results["FN"] = int(len(predictions) > 0)
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

    def custom_symmetric_similarity_classification_score_with_negative_class(
        self, predictions: list[str], references: list[str]
    ):
        best_matching_predictions = dict.fromkeys(references, 0)
        best_matching_references = dict.fromkeys(predictions, 0)

        print("Starting")
        results = {}
        for prediction in predictions:
            similarity, best_matched_reference = get_most_similar(
                prediction, references, self.string_similarity
            )
            best_matching_references[prediction] = (similarity, best_matched_reference)

        for reference in references:
            similarity, best_matching_prediction = get_most_similar(
                reference, predictions, self.string_similarity
            )
            best_matching_predictions[reference] = (
                similarity,
                best_matching_prediction,
            )

        if len(references) == 0:
            if len(predictions) == 0:
                best_matching_predictions[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_predictions[NoUseCaseOptions] = (0.0, None)

        if len(predictions) == 0:
            if len(references) == 0:
                best_matching_references[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_references[NoUseCaseOptions] = (0.0, None)

        print(best_matching_predictions, best_matching_references)

        results = {}

        unique_classes = set(best_matching_predictions.keys()).union(
            set(best_matching_references.keys())
        )

        for unique_class in unique_classes:
            if unique_class in best_matching_predictions:
                results[unique_class] = {
                    "TP": best_matching_predictions[unique_class][0],
                    "FN": 1 - best_matching_predictions[unique_class][0],
                    "best_match": best_matching_predictions[unique_class][1],
                }
            elif unique_class in best_matching_references:
                results[unique_class] = {
                    "TP": best_matching_references[unique_class][0],
                    "FP": 1 - best_matching_references[unique_class][0],
                    "best:match": best_matching_references[unique_class][1],
                }

        return results

    def custom_similarity_classification_score_with_negative_class(
        self, predictions: list[str], references: list[str]
    ):
        THRESHOLD = 0.8

        best_matching_predictions = dict.fromkeys(references, 0)
        best_matching_references = dict.fromkeys(predictions, 0)

        print("Starting")
        for prediction in predictions:
            similarity, best_matched_reference = get_most_similar(
                prediction, references, self.string_similarity
            )
            best_matching_references[prediction] = (similarity, best_matched_reference)
        for reference in references:
            similarity, best_matching_prediction = get_most_similar(
                reference, predictions, self.string_similarity
            )
            best_matching_predictions[reference] = (
                similarity,
                best_matching_prediction,
            )

        if len(references) == 0:
            if len(predictions) == 0:
                best_matching_predictions[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_predictions[NoUseCaseOptions] = (0.0, None)

        if len(predictions) == 0:
            if len(references) == 0:
                best_matching_references[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_references[NoUseCaseOptions] = (0.0, None)

        print(best_matching_predictions, best_matching_references)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        # results["TP"] = (
        #     len(
        #         [
        #             similarity
        #             for similarity, _ in best_matching_predictions.values()
        #             if similarity >= THRESHOLD
        #         ]
        #     ),
        #     mean(
        #         [
        #             (1 - similarity)
        #             for similarity, _ in best_matching_predictions.values()
        #             if similarity >= THRESHOLD
        #         ]
        #     ),
        # )
        # results["FP"] = (
        #     len(
        #         [
        #             similarity
        #             for similarity, _ in best_matching_references.values()
        #             if similarity < THRESHOLD
        #         ]
        #     ),
        #     mean(
        #         [
        #             (similarity)
        #             for similarity, _ in best_matching_references.values()
        #             if similarity < THRESHOLD
        #         ]
        #     ),
        # )
        # results["FN"] = (
        #     len(
        #         [
        #             similarity
        #             for similarity, _ in best_matching_predictions.values()
        #             if similarity < THRESHOLD
        #         ]
        #     ),
        #     mean(
        #         [
        #             (similarity)
        #             for similarity, _ in best_matching_predictions.values()
        #             if similarity < THRESHOLD
        #         ]
        #     ),
        # )
        results["TN"] = math.inf

        results["TP"] = (
            sum(
                [similarity for similarity, _ in best_matching_predictions.values()]
                + [similarity for similarity, _ in best_matching_references.values()]
            )
            / 2
        )
        results["FN"] = sum(
            [1 - similarity for similarity, _ in best_matching_predictions.values()]
        )
        results["FP"] = sum(
            [1 - similarity for similarity, _ in best_matching_references.values()]
        )

        return results


class Metrics:
    """binary score details: https://en.wikipedia.org/wiki/Precision_and_recall"""

    individual_metrics = [
        "custom_precision",
        "custom_recall",
        "custom_f1_score_min",
        "custom_f1_score",
        "custom_f1_score_max",
        "custom_f1_score_ak",
    ]

    SCORE_ALL_TOKEN = "[ALL]"

    def __init__(
        self,
        data,
        pred_id: str,
        ref_id: str,
        string_similiarity=DEFAULT_STRING_SIMILARITY,
        agg=DEFAULT_AGG,
        threshold=DEFAULT_NLP_THRESHOLD,
        skip_base_metrics=True,  # TODO: integrate ore remove these older scores
    ) -> None:
        self.data = data  # structured as in our JSON format v1
        self.pred_id = pred_id
        self.ref_id = ref_id
        self.string_similiarity = string_similiarity
        self.agg = agg
        self.threshold = threshold

        if not skip_base_metrics:
            labels = self.__ultimate_json_to_pred_and_ref(self.data)
            base_metrics = self.__base_metrics(labels, threshold)
            self.TP = base_metrics["TP"]
            self.TN = base_metrics["TN"]
            self.FP = base_metrics["FP"]
            self.FN = base_metrics["FN"]
            self.P = base_metrics["P"]
            self.N = base_metrics["N"]

    def __ultimate_json_to_pred_and_ref(
        self,
        data: dict,
    ) -> list[dict]:
        result = []

        for review in data["reviews"]:
            labels = review["labels"]
            result.append(
                {
                    "predictions": labels[self.pred_id]["usageOptions"],
                    "references": labels[self.ref_id]["usageOptions"],
                }
            )

        return result

    def __base_metrics(self, labels, threshold):
        """calculates amounts of true/false positives/negatives on label level"""
        assert 0 <= threshold <= 1

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        p = 0
        n = 0

        for label in labels:
            turker_labels, golden_labels = (
                label["predictions"],
                label["references"],
            )
            if len(golden_labels) == len(turker_labels) == 0:
                tn += 1
                n += 1
            else:
                turker_labels_copy = turker_labels.copy()
                p += len(golden_labels)
                matches = {}
                for golden_label in golden_labels:
                    matches[golden_label] = []
                    similarity, most_similar_turker_label = get_most_similar(
                        golden_label, turker_labels_copy
                    )
                    if similarity >= threshold:
                        matches[golden_label].append(most_similar_turker_label)
                        turker_labels_copy.remove(most_similar_turker_label)

                for golden_label, matched_turker_labels in matches.items():
                    if len(matched_turker_labels) == 0:
                        fn += 1
                    else:
                        tp += 1
                fp += len(turker_labels_copy)

        return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "N": n, "P": p}

    def calculate(
        self,
        metric_names=["custom_recall", "custom_precision", "custom_f1_score"],
    ):
        """returns a tuple containing score-enriched labels and a dictionary with overall scores"""
        scores = {}

        rev_id = 0

        for review in self.data["reviews"]:
            print(rev_id)
            rev_id += 1

            labels = review["labels"]
            if self.ref_id not in labels or (
                self.pred_id not in labels and self.pred_id != self.SCORE_ALL_TOKEN
            ):
                continue  # unable to score

            for label_id in labels:
                if self.pred_id == label_id or (
                    self.pred_id == self.SCORE_ALL_TOKEN and self.pred_id != self.ref_id
                ):
                    # going deep into the tunnels :P (TODO: write this shorter/cleaner)
                    if "scores" not in labels[label_id]["metadata"]:
                        labels[label_id]["metadata"]["scores"] = {}
                    if self.ref_id not in labels[label_id]["metadata"]["scores"]:
                        labels[label_id]["metadata"]["scores"][self.ref_id] = {}

                    data_scores = labels[label_id]["metadata"]["scores"][self.ref_id]

                    pred = labels[label_id]["usageOptions"]
                    ref = labels[self.ref_id]["usageOptions"]

                    for metric_name in metric_names:
                        try:
                            metric_result = getattr(self, metric_name)(pred, ref)
                        except ZeroDivisionError:
                            continue

                        if metric_name in self.individual_metrics:
                            data_scores[metric_name] = metric_result
                        else:
                            scores[metric_name] = metric_result

        return self.data, scores

    def false_positive(self):
        return self.FP

    def false_negative(self):
        return self.FN

    def true_positive(self):
        return self.TP

    def true_negative(self):
        return self.TN

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def recall(self):
        return self.TP / (self.TP + self.FN)

    def specificity(self):
        return self.TN / (self.TN + self.FP)

    def miss_rate(self):
        return self.FN / (self.FN + self.TP)

    def fall_out(self):
        return self.FP / (self.FP + self.TN)

    def false_discovery_rate(self):
        return self.FP / (self.FP + self.TP)

    def false_omission_rate(self):
        return self.FN / (self.FN + self.TN)

    def accuracy(self):
        return (self.TP + self.TN) / (self.P + self.N)

    def balanced_accuracy(self):
        return (self.recall() + self.specificity()) / 2

    def f1(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)

    def custom_recall(self, pred, ref):
        return custom_recall(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similiarity,
            agg=mean,
        )

    def custom_precision(self, pred, ref):
        return custom_precision(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similiarity,
            agg=mean,
        )

    def custom_f1_score_min(self, pred, ref):
        return custom_f1_score(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similiarity,
            agg=min,
        )

    def custom_f1_score(self, pred, ref):
        return custom_f1_score(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similiarity,
            agg=mean,
        )

    def custom_f1_score_max(self, pred, ref):
        return custom_f1_score(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similiarity,
            agg=max,
        )

    def custom_f1_score_ak(self, pred, ref):
        return custom_f1_score_ak(
            predictions=pred,
            references=ref,
            string_similiarity=self.string_similiarity,
        )

    def bleu(self):
        return bleu_score(
            prediction=[prediction for prediction in self.labels["predictions"]],
            reference=[reference for reference in self.labels["references"]],
        )

    def sacrebleu(self):
        return sacrebleu_score(
            prediction=[prediction for prediction in self.labels["predictions"]],
            reference=[reference for reference in self.labels["references"]],
        )

    def rouge1(self):
        return rouge_score(
            prediction=[prediction for prediction in self.labels["predictions"]],
            reference=[reference for reference in self.labels["references"]],
            rouge_score="rouge1",
        )

    def rouge2(self):
        return rouge_score(
            prediction=[prediction for prediction in self.labels["predictions"]],
            reference=[reference for reference in self.labels["references"]],
            rouge_score="rouge2",
        )

    def rougeL(self):
        return rouge_score(
            prediction=[prediction for prediction in self.labels["predictions"]],
            reference=[reference for reference in self.labels["references"]],
            rouge_score="rougeL",
        )

    def rougeLsum(self):
        return rouge_score(
            prediction=[prediction for prediction in self.labels["predictions"]],
            reference=[reference for reference in self.labels["references"]],
            rouge_score="rougeLsum",
        )
