from evaluation.scoring.core import get_most_similar
from evaluation.scoring.standard_metrics import bleu_score, sacrebleu_score, rouge_score
from evaluation.scoring.custom_metrics import (
    custom_f1_score,
    custom_recall,
    custom_precision,
    custom_f1_score_ak,
)
from statistics import mean

DEFAULT_STRING_SIMILARITY = "all-mpnet-base-v2"
DEFAULT_AGG = mean
DEFAULT_NLP_THRESHOLD = 0.7


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
