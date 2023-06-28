import argparse
from t5_kernl import OptimizedKernlModel
from experiments.inference_benchmarks.t5_default import TransformersDefault
from experiments.inference_benchmarks.t5_tf32 import TransformersTF32
from experiments.inference_benchmarks.t5_mixed_int8 import MixedInt8
import pandas as pd
import datetime
from helpers.review_set import ReviewSet


def current_timestamp() -> str:
    timestamp = datetime.datetime.now().timestamp()

    # Convert the timestamp to a datetime object
    dt = datetime.datetime.fromtimestamp(timestamp)

    # Convert the datetime object to a string representation
    str_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    return str_time


def main(args):
    # Access the values of the input options
    print(args)
    num_samples = args.num_samples
    inference_modes = args.inference_modes
    batch_size = args.batch_size
    input_ids_length = args.input_ids_length
    max_output_length = args.max_output_length
    review_set = ReviewSet.from_files(args.review_set)

    input = []
    for review in review_set:
        input.append(review.get_prompt())

    print(f"Bechmarking with {len(input)} examples")
    models = []
    model_kwargs = {"max_length": 25}

    for inference_mode in inference_modes:
        if inference_mode == "Kernl":
            models.append(OptimizedKernlModel(num_samples, batch_size))
        elif inference_mode == "Default":
            models.append(TransformersDefault(num_samples, batch_size))
        elif inference_mode == "TF32":
            models.append(TransformersTF32(num_samples, batch_size))
        elif inference_mode == "mixed_int8":
            models.append(MixedInt8(num_samples, batch_size))
        else:
            raise ValueError(f"Inference mode: {inference_mode} not known")

    results = []
    for model in models:
        optimization_time, _ = model.bechmark_initialization()
        tokenization_time, tokenization_std = model.benchmark_tokenize(input)
        run_time, run_std = model.benchmark(model_kwargs)
        results.append(
            {
                "optimization_time": optimization_time,
                "tokenization_mean": tokenization_time,
                "tokenization_std": tokenization_std,
                "run_mean": run_time,
                "run_std": run_std,
                "n_samples": num_samples,
            }
        )

    print(results)

    pd.DataFrame.from_records(results).to_csv(
        f"benchmark_results-{current_timestamp()}.csv"
    )


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Script description")

    # Add the input options
    parser.add_argument("review_set", help="Review set to benchmark on")
    parser.add_argument("inference_modes", nargs="+", help="Inference modes")
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples (default: 10)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=512 + 64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "-i",
        "--input-ids-length",
        type=int,
        default=50,
        help="Input IDs length (default: 50)",
    )
    parser.add_argument(
        "-o",
        "--max-output-length",
        type=int,
        default=20,
        help="Max output length (default: 20)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
