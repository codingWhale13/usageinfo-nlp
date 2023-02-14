import argparse
import json
from pathlib import Path

from core import gpt_predictions_to_labels
from metrics import Metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="JSON file containing predictions from GPT prompts",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        help="folder to save plots (default: current directory)",
    )
    args = parser.parse_args()

    # set save path to current directory if not specified
    save_path = args.save_path if args.save_path is not None else Path().absolute()
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # file name without extension will be the prefix for plot file names
    base_name = Path(args.predictions).stem

    # compute scores
    labels = gpt_predictions_to_labels(args.predictions)
    labels, _ = Metrics(labels).calculate()

    # save computed scores to file
    with open(Path(save_path, f"{base_name}_scores.json"), "w") as json_file:
        json.dump(labels, json_file)
