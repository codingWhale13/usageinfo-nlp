import spacy
from .worker_custom_metrics import extract_labels
from .worker_custom_metrics import get_most_similar
from .worker_custom_metrics import custom_precision, custom_recall

"""
INSTALL spacy model
python -m spacy download en_core_web_md
"""

DEFAULT_NLP_THRESHOLD = 0.7


class Metrics:
    """Score details: https://en.wikipedia.org/wiki/Precision_and_recall"""

    def __init__(
        self, worker_labels, golden_labels, threshold=DEFAULT_NLP_THRESHOLD
    ) -> None:
        self.labels = extract_labels(worker_labels, golden_labels)
        self.nlp = spacy.load("en_core_web_md")
        self.threshold = threshold
        results = self.__base_metrics(self.labels, self.nlp, threshold)

        self.TP = results["TP"]
        self.TN = results["TN"]
        self.FP = results["FP"]
        self.FN = results["FN"]
        self.P = results["P"]
        self.N = results["N"]

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
        return custom_recall(self.labels, self.nlp)

    def custom_precision(self):
        return custom_precision(self.labels, self.nlp)

    def calculate(self, metric_names):
        scores = {}
        for metric_name in metric_names:
            try:
                metric_function = getattr(self, metric_name)
                scores[metric_name] = metric_function()
            except ZeroDivisionError:
                scores[metric_name] = None
        return scores

    def __base_metrics(self, labels, nlp, threshold):
        """calculates amounts of true/false positives/negatives on label level"""

        assert 0 <= threshold <= 1

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        p = 0
        n = 0

        for turker_labels, golden_labels in labels:
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
                        golden_label, turker_labels_copy, nlp
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
