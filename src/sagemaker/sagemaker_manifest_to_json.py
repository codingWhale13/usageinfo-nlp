#!/usr/bin/env python3

import argparse
import os
import json
from pathlib import Path
from typing import Union
from src.sagemaker.manifest_to_json_backend import extract_json_from_manifest


def get_run_name_from_manifest_file(manifest_file_path):
    with open(manifest_file_path, "r") as manifest:
        line = json.loads(manifest.readline())
        for key, value in line.items():
            if isinstance(value, dict) and "job-name" in value:
                return line[key]["job-name"]


def convert_manifest_output_to_json(
    input_path: Union[Path, str], vendor_name: str = "vendor"
):
    run_name = get_run_name_from_manifest_file(input_path)

    with open(input_path, "r") as manifest:
        reviews = extract_json_from_manifest(
            manifest, run_name, vendor_name=vendor_name
        )

        output_path = f"{os.path.splitext(os.path.abspath(input_path))[0]}.json"
        with open(output_path, "w") as output_file:
            json.dump(reviews, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_file", help="manifest sagemaker output file")
    parser.add_argument("--vendor-name", "-v", default="vendor", help="vendor name")

    args = parser.parse_args()

    convert_manifest_output_to_json(args.manifest_file, args.vendor_name)
