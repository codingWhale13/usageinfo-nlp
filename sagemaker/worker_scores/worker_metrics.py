from worker_custom_metrics import extract_labels
from sentence_transformers import SentenceTransformer, util
import os 
import sys
path = os.path.dirname(os.path.realpath(__file__))
new_path_split = path.split(os.sep)[:-2]
sys.path.append(os.path.join(os.path.sep, *new_path_split))

from utils.extract_reviews import extract_reviews_with_usage_options_from_json



"""
INSTALL spacy model
python -m spacy download en_core_web_md
"""

DEFAULT_NLP_THRESHOLD = 0.7



class Metrics:
    """Score details: https://en.wikipedia.org/wiki/Precision_and_recall"""

    def __init__(
        self, worker_labels, golden_labels, threshold=DEFAULT_NLP_THRESHOLD, model_checkpoint='all-MiniLM-L6-v2'
    ) -> None:
        if type(worker_labels) == str:
            worker_labels = extract_reviews_with_usage_options_from_json(worker_labels)
        if type(golden_labels) == str:
            golden_labels = extract_reviews_with_usage_options_from_json(golden_labels)

        self.labels = extract_labels(worker_labels, golden_labels)
        self.model = SentenceTransformer(model_checkpoint)
        self.threshold = threshold
        results = self.__base_metrics()

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

    def calculate(self, metric_names):
        scores = {'N': self.N, 'P': self.P}
        for metric_name in metric_names:
            try:
                metric_function = getattr(self, metric_name)
                scores[metric_name] = metric_function()
            except ZeroDivisionError:
                scores[metric_name] = None
        return scores

    def __cosine_similar(self,str_1, str_2):
        #Compute embeddings
        embeddings = self.model.encode([str_1, str_2], convert_to_tensor=True)
        return  util.cos_sim(embeddings[0], embeddings[1])


    def __get_most_similar(self, label, options) -> tuple[float, str]:
        """For a single `label`, find the most similar match from `options`.

        Returns tuple (best similarity score, option with best similiarity score)."""
        assert 0 <= self.threshold <= 1

        result = (0, None)
        for option in options:
            similarity = self.__cosine_similar(option, label)
            if similarity >= max(result[0], self.threshold):
                result = (similarity, option)

        return result

    def __base_metrics(self):
        """calculates amounts of true/false positives/negatives on label level"""

        assert 0 <= self.threshold <= 1

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        p = 0
        n = 0

        for turker_labels, golden_labels in self.labels:
            if len(golden_labels) == len(turker_labels) == 0:
                tn += 1
                n += 1
            else:
                p += len(golden_labels)
                matches = {}
                for golden_label in golden_labels:
                    matches[golden_label] = []

                not_matched_turker_labels = []
                
                for turker_label in turker_labels:
                    similarity, most_similar_golden_label = self.__get_most_similar(
                        turker_label, golden_labels
                    )

                    if similarity >= self.threshold:
                        matches[most_similar_golden_label].append(turker_label)
                    else:
                        not_matched_turker_labels.append(turker_label)
                

                for golden_label, matched_turker_labels in matches.items():
                    if len(matched_turker_labels) == 0:
                        fn += 1
                    else:
                        tp += 1
                fp += len(not_matched_turker_labels) 



        return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "N": n, "P": p}
