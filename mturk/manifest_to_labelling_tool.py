import json
from pathlib import Path
from typing import Union

def extract_json_from_manifest(input_path: Union[Path, str], output_path: Union[Path, str] = None):
    with open(input_path, "r") as turker_labels:
        reviews = {"reviews": [], "maxReviewIndex": 0}
        run_name = get_run_name(turker_labels)
        if output_path is None:
            output_path = f"{run_name}-output.json"
        for line in turker_labels:
            data = json.loads(line)
            review_bodies = json.loads(data["source"])
            number_of_workers_per_hit = len(data[run_name]["annotationsFromAllWorkers"])
            number_of_reviews_per_hit = len(review_bodies)
            for worker in range(number_of_workers_per_hit):
                for review in range(number_of_reviews_per_hit):
                    inner_annotations = json.loads(data[run_name]["annotationsFromAllWorkers"][worker]["annotationData"]["content"])
                    annotations = json.loads(inner_annotations["annotations"])["annotations"]
                    customUsageOptions = json.loads(inner_annotations["annotations"])["customUsageOptions"]
                    reviews["reviews"].append(data["metadata"][review] | {
                    "review_body": review_bodies[review],
                    "label": {
                        "isFlagged": False,
                        "annotations": annotations[review],
                        "customUsageOptions": customUsageOptions[review],
                        "replacementClasses": {}
                    },
                    "inspectionTime": None})
        with open(output_path, "w") as output_file:
            json.dump(reviews, output_file)

def get_run_name(turker_labels):
    data = json.loads(turker_labels.readline())
    for key, value in data.items():
        if isinstance(value, dict) and "job-name" in value:
            return data[key]["job-name"]


extract_json_from_manifest('example_data/output-test-run-cogito.manifest')