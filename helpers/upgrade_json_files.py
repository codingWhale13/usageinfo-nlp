import copy
from datetime import datetime
import json
from typing import Union
from pathlib import Path
from review_set import REVIEW_ATTRIBUTES

review_fields_other_than_labels = REVIEW_ATTRIBUTES.remove("labels")


def v0_to_v1(
    v0_path: Union[str, Path],
    v1_path: Union[str, Path],
    label_id: str,
    source: str = "labellingTool",
):
    """Upgrades a JSON file in "v0" format (pre March 2023) to our more consistent v1 format

    Args:
    `label_id`: unique id for where the labels comes from, like "golden_v2" or "davinci_freddy_v1"
    `source` can be either "labellingTool" or "openAI" and will be used as key for metadata
    """

    with open(v0_path) as file_v0:
        data_v0 = json.load(file_v0)
        data_v1 = {"version": 1, "reviews": []}

        for review_v0 in data_v0["reviews"]:
            review_v1 = {}
            review_metadata = {}

            for key in review_v0:
                if key in review_fields_other_than_labels:
                    review_v1[key] = review_v0[key]
                elif key == "label":
                    if source == "labellingTool":
                        usage_options = review_v0[key]["customUsageOptions"] + [
                            " ".join(annotation["tokens"])
                            for annotation in review_v0[key]["annotations"]
                        ]
                        for info in review_v0[key]:
                            if (
                                info != "replacementClasses"
                            ):  # this is a relict of the past...
                                review_metadata[info] = review_v0[key][info]
                    elif source == "openAI":
                        usage_options = review_v0[key]["usageOptions"]  # I guess?
                    else:
                        raise ValueError(
                            f"expected parameter 'source' to be labellingTool or openAI but got {source}"
                        )
                    review_v1["labels"] = {
                        label_id: {"usageOptions": usage_options, "metadata": {}}
                    }
                else:
                    review_metadata[key] = review_v0[key]

            review_v1["labels"][label_id]["metadata"][source] = review_metadata
            data_v1["reviews"].append(review_v1)

        with open(v1_path, "w") as file_v1:
            json.dump(data_v1, file_v1)


def v1_to_v2(
    v1_path: Union[str, Path],
    v2_path: Union[str, Path],
):
    """Upgrades a JSON file in v1 format to the improved v2 format"""

    timestamp = datetime.now().astimezone().isoformat()  # using ISO 8601

    with open(v1_path) as file_v1:
        data_v1 = json.load(file_v1)
        data_v2 = {"version": 2, "reviews": {}}

        for review_v1 in data_v1["reviews"]:
            review_id = review_v1["review_id"]

            review_v2 = copy.deepcopy(review_v1)
            del review_v2["review_id"]  # the review id will be the key

            for label_id in review_v2["labels"]:
                scores = review_v2["labels"][label_id]["metadata"].get("scores", {})
                review_v2["labels"][label_id]["scores"] = scores
                datasets = review_v2["labels"][label_id]["metadata"].get("datasets", [])
                review_v2["labels"][label_id]["datasets"] = datasets
                review_v2["labels"][label_id]["createdAt"] = timestamp

            data_v2["reviews"][review_id] = review_v2
        with open(v2_path, "w") as file_v2:
            json.dump(data_v2, file_v2)
