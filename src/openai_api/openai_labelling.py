#!/usr/bin/env python3
import argparse
import asyncio
import json
import os

import dotenv

from src.review_set import ReviewSet
from src.helpers.worker import Worker
from src.openai_api.openai_backend import CHAT_MODELS, MODEL_NAME_MAPPING, generate_label

dotenv.load_dotenv()


DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_LOG_PROBS = 5
DEFAULT_SAVE_PERCENTAGE = 5
DEFAULT_WORKERS = 4

MAXIMUM_LABELS_WITHOUT_SAVING = 500


def parse_args():
    def restricted_temperature(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

        if x < 0.0 or x > 2.0:
            raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 2.0]")
        return x

    arg_parser = argparse.ArgumentParser(
        description="Let an OpenAI model label the reviews in a json file."
    )
    arg_parser.add_argument("--file", "-f", required=True, help="json file path")
    arg_parser.add_argument(
        "--prompt-id",
        "-p",
        default="2_shot",
        help="Prompt to use for openai labelling",
    )
    arg_parser.add_argument(
        "-i",
        "--label_id_suffix",
        default="",
        help="Unique identifier for the new label",
    )
    arg_parser.add_argument(
        "-w",
        "--workers",
        default=DEFAULT_WORKERS,
        type=int,
        help="Number of workers to use",
    )
    arg_parser.add_argument(
        "--save-percentage",
        "-s",
        choices=range(1, 101),
        type=int,
        default=DEFAULT_SAVE_PERCENTAGE,
        help="Make an intermediate save every n percent of processed reviews",
        metavar="[0-100]",
    )
    arg_parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, help="OpenAI model to use"
    )
    arg_parser.add_argument(
        "--temperature",
        "-t",
        type=restricted_temperature,
        default=DEFAULT_TEMPERATURE,
        help="Set the temperature for the sampling",
        metavar="[0.0-2.0]",
    )
    arg_parser.add_argument(
        "--log-probs",
        "-l",
        choices=range(1, 6),
        type=int,
        default=DEFAULT_LOG_PROBS,
        help="Define how many logprobs you want to save",
        metavar="[1-5]",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def label_review_set(
    review_set: ReviewSet,
    prompt_id: str,
    label_id_suffix: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    log_probs: int = DEFAULT_LOG_PROBS,
    workers: int = DEFAULT_WORKERS,
    save_percentage: int = DEFAULT_SAVE_PERCENTAGE,
) -> None:
    print(
        prompt_id,
        label_id_suffix,
        model,
        temperature,
        log_probs,
        workers,
        save_percentage,
    )
    asyncio.run(
        label_review_set_async(
            review_set,
            prompt_id,
            label_id_suffix,
            model,
            temperature,
            log_probs,
            workers,
            save_percentage,
        )
    )


async def label_review_set_async(
    review_set: ReviewSet,
    prompt_id: str,
    label_id_suffix: str,
    model: str,
    temperature: float,
    log_probs: int,
    workers: int,
    save_percentage: int,
) -> None:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    prompt_file_name = f"{script_dir}/prompts.json"

    with open(prompt_file_name, "r") as prompt_file:
        prompts_json = json.load(prompt_file)
        prompt_type = "chat" if model in CHAT_MODELS else "text"
        if prompt_id not in prompts_json[prompt_type].keys():
            exit(
                f"Prompt {prompt_id} of type {prompt_type} not found in {prompt_file_name}."
            )
        prompt = prompts_json[prompt_type][prompt_id]["prompt"]

    label_id = "{model_name}-{prompt_id}{hyphen}{id}".format(
        model_name=MODEL_NAME_MAPPING[model],
        prompt_id=prompt_id,
        id=label_id_suffix,
        hyphen="-" if label_id_suffix != "" else "",
    )

    num_reviews = len(review_set)
    intermediate_save_size = min(
        max(1, int(num_reviews * (save_percentage / 100))),
        MAXIMUM_LABELS_WITHOUT_SAVING,
    )
    count = 0

    overwrite_label = False
    if label_id in review_set.get_all_label_ids():
        if input(f"Label {label_id} already exists. Overwrite? (y/N): ").lower() != "y":
            exit("Aborted.")
        overwrite_label = True

    def generate_label_worker_item(review_id: str, review: dict):
        def task():
            nonlocal count
            usageOptions, metadata = generate_label(
                review,
                prompt,
                model=model,
                temperature=temperature,
                logprobs=log_probs,
                prompt_id=prompt_id,
            )
            review_set[review_id].add_label(
                label_id=label_id,
                usage_options=usageOptions,
                metadata=metadata,
                overwrite=overwrite_label,
            )

            if count % intermediate_save_size == 0 and count != 0:
                review_set.save()
                print(
                    f"{count}/{num_reviews} reviews processed. Intermediate results saved. 🚀"
                )

            count += 1

        return task

    labelling_queue = asyncio.Queue()

    for review_id, review in review_set.reviews.items():
        labelling_queue.put_nowait(generate_label_worker_item(review_id, review))

    worker = Worker(labelling_queue, n=workers)
    print(
        f"Setup queue with {worker.queue.qsize()} reviews. Starting {worker.n} workers in parallel 🚀"
    )

    await worker.run()

    print("All reviews processed. 🌪️🌪️🌪️")
    review_set.save()


def main():
    args, _ = parse_args()
    review_set = ReviewSet.from_files(os.path.abspath(args.file))

    args = vars(args)
    del args["file"]
    label_review_set(review_set, **args)


if __name__ == "__main__":
    main()
