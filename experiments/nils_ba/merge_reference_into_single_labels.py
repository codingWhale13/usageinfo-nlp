import os
import subprocess

from helpers.review_set import ReviewSet

FOLDER_PATH = "data/ba/single_label/exp_hp"
REF_LABEL_ID = "chat_gpt_best"

directory = os.fsencode(FOLDER_PATH)
for json_file in os.listdir(directory):
    filename = os.fsdecode(json_file)
    assert filename.endswith(".json")

    if filename == REF_LABEL_ID:
        continue

    label_id = filename.split(".json")[0]

    rs = ReviewSet.from_files(
        os.path.join(FOLDER_PATH, f"{REF_LABEL_ID}.json"),
        os.path.join(FOLDER_PATH, f"{label_id}.json"),
    )
    rs.save(os.path.join(FOLDER_PATH, f"{label_id}.json"))
