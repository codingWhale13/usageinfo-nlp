import io
import json
import boto3
from datetime import datetime


def consolidate_labels(label_candidates: dict):
    return list(label_candidates.values())[0]


def extract_json_from_manifest(
    manifest: io.IOBase, run_name: str = None, vendor_name: str = "vendor"
):
    reviewset = {"version": 5, "reviews": {}}

    for line in manifest:
        data = json.loads(line)
        review_bodies = json.loads(data["source"])

        for review_count, review_body in enumerate(review_bodies):
            review_id = data["metadata"][review_count]["review_id"]
            reviewset["reviews"][review_id] = data["metadata"][review_count]
            reviewset["reviews"][review_id]["review_body"] = review_body
            reviewset["reviews"][review_id]["labels"] = {}
            reviewset["reviews"][review_id].pop("review_id", None)
            reviewset["reviews"][review_id].pop("customUsageOptions", None)
            reviewset["reviews"][review_id].pop("annotations", None)

            label_candidates = {}
            for worker_count, worker in enumerate(
                data[run_name]["annotationsFromAllWorkers"]
            ):
                worker_id = worker["workerId"]
                annotation_data = json.loads(worker["annotationData"]["content"])
                annotation_data = json.loads(annotation_data["annotations"])

                label_candidates[worker_id] = {
                    "createdAt": datetime.now().astimezone().isoformat(),
                    "usageOptions": annotation_data["customUsageOptions"][review_count]
                    + [
                        " ".join(section["tokens"])
                        for section in annotation_data["annotations"][review_count]
                    ],
                    "scores": {},
                    "datasets": [],
                    "augmentations": {},
                    "metadata": {
                        "labellingTool": {
                            key: value[review_count]
                            for key, value in annotation_data.items()
                        }
                    },
                }

            reviewset["reviews"][review_id]["labels"] = {
                f"aws-{vendor_name}-{run_name}": consolidate_labels(label_candidates)
            }

    return reviewset


def lambda_handler(event, context):
    s3 = boto3.resource("s3")
    print(f"Event JSON for debugging purpose: {event}")

    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    output_manifest_key = event["Records"][0]["s3"]["object"]["key"]
    print(
        f"Function was triggered for creation of object with key: {output_manifest_key} in bucket: {bucket}"
    )

    labelling_job_name = output_manifest_key.split("/")[1]
    print(f"Identified Labelling job name: {labelling_job_name}")

    output_manifest_object = s3.Object(bucket, output_manifest_key)
    output_manifest = output_manifest_object.get()["Body"].read()

    output_json = extract_json_from_manifest(
        io.BytesIO(output_manifest), labelling_job_name
    )

    output_json_key = f"output-sanitized/{labelling_job_name}.json"
    output_json_object = s3.Object(bucket, output_json_key)
    output_json_object.put(Body=json.dumps(output_json).encode("utf-8"))
