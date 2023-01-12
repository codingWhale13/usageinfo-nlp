import os
import json
from pathlib import Path
from typing import Union
import argparse
from sagemaker_manifest_to_json_lambda import extract_json_from_manifest


def get_run_name_from_manifest_file(manifest_file_path):
    with open(manifest_file_path, "r") as manifest:
        line = json.loads(manifest.readline())
        for key, value in line.items():
            if isinstance(value, dict) and "job-name" in value:
                return line[key]["job-name"]

def convert_manifest_output_to_json(input_path: Union[Path, str]):
    run_name = get_run_name_from_manifest_file(input_path)

    with open(input_path, "r") as manifest:
        reviews = extract_json_from_manifest(manifest, run_name)

        output_path = f"{os.path.dirname(os.path.abspath(input_path))}/{run_name}.json"
        with open(output_path, "w") as output_file:
            json.dump(reviews, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_file",  help=".manifest sagemaker output file")

    args = parser.parse_args()

    convert_manifest_output_to_json(args.manifest_file)
