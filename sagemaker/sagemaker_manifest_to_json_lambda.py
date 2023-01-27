import io
import json
import boto3


def extract_json_from_manifest(manifest: io.IOBase, run_name: str = None):
    reviews = {"reviews": [], "maxReviewIndex": 0}

    for line in manifest:
        data = json.loads(line)
        review_bodies = json.loads(data["source"])
        number_of_workers_per_hit = len(data[run_name]["annotationsFromAllWorkers"])
        number_of_reviews_per_hit = len(review_bodies)

        for worker in range(number_of_workers_per_hit):
            for review in range(number_of_reviews_per_hit):
                worker_id = data[run_name]["annotationsFromAllWorkers"][worker]["workerId"]
                inner_annotations = json.loads(data[run_name]["annotationsFromAllWorkers"][worker]["annotationData"]["content"])
                labelling_data = json.loads(inner_annotations["annotations"])

                annotations = labelling_data["annotations"]
                customUsageOptions = labelling_data["customUsageOptions"]
                worker_inspection_time = None
                if "inspectionTimes" in labelling_data:
                    worker_inspection_time = labelling_data["inspectionTimes"][review]

                reviews["reviews"].append(data["metadata"][review] | {
                "review_body": review_bodies[review],
                "label": {
                    "isFlagged": False,
                    "annotations": annotations[review],
                    "customUsageOptions": customUsageOptions[review]
                },
                "workerId": worker_id,
                "workerInspectionTime": worker_inspection_time,
                "inspectionTime": None})

    return reviews

def lambda_handler(event, context):
    s3 = boto3.resource('s3')
    print(f"Event JSON for debugging purpose: {event}")

    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    output_manifest_key = event["Records"][0]["s3"]["object"]["key"]
    print(f"Function was triggered for creation of object with key: {output_manifest_key} in bucket: {bucket}")

    labelling_job_name = output_manifest_key.split("/")[1]
    print(f"Identified Labelling job name: {labelling_job_name}")

    output_manifest_object = s3.Object(bucket, output_manifest_key)
    output_manifest = output_manifest_object.get()['Body'].read()

    output_json = extract_json_from_manifest(io.BytesIO(output_manifest), labelling_job_name)

    output_json_key = f"output-sanitized/{labelling_job_name}.json"
    output_json_object = s3.Object(bucket, output_json_key)
    output_json_object.put(Body=json.dumps(output_json).encode('utf-8'))