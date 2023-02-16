from evaluation.scoring.core import get_most_similar
from evaluation.scoring.standard_metrics import bleu_score, sacrebleu_score, rouge_score
from evaluation.scoring.custom_metrics import (
    custom_f1_score,
    custom_recall,
    custom_precision,
)
from statistics import mean

DEFAULT_STRING_SIMILARITY = "all-mpnet-base-v2"
DEFAULT_AGG = mean
DEFAULT_NLP_THRESHOLD = 0.7


class Metrics:
    """binary score details: https://en.wikipedia.org/wiki/Precision_and_recall"""

    individual_metrics = ["custom_recall", "custom_precision", "custom_f1_score"]

    def __init__(
        self,
        labels: list[dict],
        string_similiarity=DEFAULT_STRING_SIMILARITY,
        agg=DEFAULT_AGG,
        threshold=DEFAULT_NLP_THRESHOLD,
    ) -> None:
        self.labels = labels
        self.string_similiarity = string_similiarity
        self.agg = agg
        self.threshold = threshold

        base_metrics = self.__base_metrics(self.labels, threshold)
        self.TP = base_metrics["TP"]
        self.TN = base_metrics["TN"]
        self.FP = base_metrics["FP"]
        self.FN = base_metrics["FN"]
        self.P = base_metrics["P"]
        self.N = base_metrics["N"]

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

        for label in self.labels:
            if "scores" not in label:
                label["scores"] = {}

        for metric_name in metric_names:
            try:
                metric_results = getattr(self, metric_name)()
            except ZeroDivisionError:
                continue

            if metric_name in self.individual_metrics:
                for idx, label in enumerate(self.labels):
                    label["scores"][metric_name] = metric_results[idx]
                scores[metric_name] = mean(
                    [label["scores"][metric_name] for label in self.labels]
                )
            else:
                scores[metric_name] = metric_results

        return self.labels, scores

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

    def custom_recall(self):
        return [
            custom_recall(
                predictions=label["predictions"],
                references=label["references"],
                string_similiarity=self.string_similiarity,
                agg=mean,
            )
            for label in self.labels
        ]

    def custom_precision(self):
        return [
            custom_precision(
                predictions=label["predictions"],
                references=label["references"],
                string_similiarity=self.string_similiarity,
                agg=mean,
            )
            for label in self.labels
        ]

    def custom_min_precision(self):
        return [
            custom_precision(
                predictions=label["predictions"],
                references=label["references"],
                string_similiarity=self.string_similiarity,
                agg=min,
            )
            for label in self.labels
        ]

    def custom_f1_score(self):
        return [
            custom_f1_score(
                predictions=label["predictions"],
                references=label["references"],
                string_similiarity=self.string_similiarity,
                agg=mean,
            )
            for label in self.labels
        ]

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
