import os
from copy import deepcopy
from src.review_set import ReviewSet

# This script takes a JSON file with many labels and splits it into one file per label.

ALL_LABELS_PATH = "data/ba/ba-30k-test-all.json"
SINGLE_LABELS_FOLDER = "data/ba/single_label"

large_review_set = ReviewSet.from_files(ALL_LABELS_PATH)

label_ids = large_review_set.get_all_label_ids()
single_review_sets = {
    label_id: ReviewSet(version=5, reviews={}) for label_id in label_ids
}

for review_id, review in large_review_set.items():
    for label_id in label_ids:
        if label_id in review.data["labels"]:
            review_copy = deepcopy(review)
            review_copy.data["labels"] = {
                label_id: review_copy.data["labels"][label_id]
            }
            single_review_sets[label_id].reviews[review_id] = review_copy

for label_id in label_ids:
    single_review_sets[label_id].save(
        os.path.join(SINGLE_LABELS_FOLDER, f"{label_id}.json")
    )
