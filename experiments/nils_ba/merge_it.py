import os

from helpers.review_set import ReviewSet

PATH_IN = "experiments/nils_ba/experiment_compress/data/30k-val.json"
PATH_OUT = "experiments/nils_ba/experiment_compress/data/30k-val.json"
NEW_LABEL_ID = "chat_gpt_clean"

rs = ReviewSet.from_files(PATH_IN)
rs.merge_labels(
    "bp-chat_gpt_correction", "chat_gpt-vanilla-baseline", new_label_id=NEW_LABEL_ID
)

rs.save(PATH_OUT)
