from evaluation.scoring.core import get_similarity
import pickle
from evaluation.scoring.evaluation_cache import EvaluationCache
import os
import pickle
from statistics import median
from statistics import mean
import argparse
import json
from pathlib import Path

CUSTOM_SIMILARITIES = [
    "avetis",
    "nils",
    "matthis",
    "toller_hecht_17",
    "leo_h",
    "frederic",
]

METRICS1 = ["bleu", "sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum"]
METRICS2 = ["all-mpnet-base-v2", "spacy", "sentence-t5-xxl", "gtr-t5-xxl"]
COMPS = ["cosine", "euclidean", "cosine_relu"]

def parse_args():
    parser = argparse.ArgumentParser("This script will preprocess and combine the sidequest results and store them in a .pkl and .json format.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the input files directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=Path(__file__).resolve().parent,
        help="Path to the directory in which combined/preprocessed .pkl and .json will be stored."
    )
    return parser.parse_args() 


def main():
    args = parse_args()
    
    res = {} 
    
    for file in Path(args.input).iterdir():
        if file.suffix == ".pkl":
            with open(file, "rb") as f:
                data = pickle.load(f)
                for key, value in data.items():
                    if key in res:
                        res[key]["similarities"] |= value["similarities"]
                    else:
                        res[key] = value

    for value in res.values():
        value["similarities"]["bp_median"] = median(
            [
                value2
                for key, value2 in value["similarities"].items()
                if key in CUSTOM_SIMILARITIES
            ]
        )
        value["similarities"]["bp_mean"] = mean(
            [
                value2
                for key, value2 in value["similarities"].items()
                if key in CUSTOM_SIMILARITIES
            ]
        )

    for value in res.values():
        for metric in list(value["similarities"].keys()):
            if metric not in METRICS1 + ["bp_median", "bp_mean"]:
                del value["similarities"][metric]


    for comp in COMPS:
        for pair, value in res.items():
            for metric in METRICS2:
                value["similarities"][metric + "_" + comp] = get_similarity(
                    pair[0], pair[1], comparator=metric, distance_metric=comp
                )

    for pair, value in res.items():
        value["similarities"]["openai_nils"] = get_similarity(
            pair[0], pair[1], comparator="openai"
        )

        OPENAI_SIM_PARAMS = {
            "model": "gpt-3.5-turbo",
            "prompt_id": "avetis_v1",
            "temperature": 1.0,
        }
        value["similarities"]["openai_avetis"] = get_similarity(
            pair[0], pair[1], comparator="openai", openai_params=OPENAI_SIM_PARAMS
        )

    
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    with open(Path(args.output) / "pairs_preprocessed.pkl", "wb") as f:
        pickle.dump(res, f)

    for pair in list(res.keys()):
        res["; ".join(pair)] = res[pair]
        del res[pair]   
    

    with open(Path(args.output) / "pairs_preprocessed.json", "w") as f:
        json.dump(res, f)


    EvaluationCache.get().save_to_disk()


if __name__ == "__main__":
    main()
