import argparse
import os
import asyncio
import json
from datetime import datetime, timezone
import openai
import random
from copy import deepcopy

from openai_api.openai_labelling import (
    DEFAULT_MODEL,
    MODEL_NAME_MAPPING,
)
from helpers.worker import Worker
from openai_api.openai_backend import request_openai_api

REVIEWSET_VERSION = 4
BASE_PATH = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
with open(
    f"{BASE_PATH}/data_exploration/good_to_label_categories.txt",
    "r",
) as file:
    PRODUCT_CATEGORIES = [category.strip() for category in file.readlines()]


def get_default_review(label_id):
    return {
        "marketplace": f"{label_id}",
        "customer_id": f"{label_id}",
        "product_id": f"{label_id}",
        "product_parent": f"{label_id}",
        "product_title": "",
        "product_category": f"{label_id}",
        "star_rating": 0,
        "helpful_votes": 0,
        "total_votes": 0,
        "vine": 0,
        "verified_purchase": 0,
        "review_headline": "",
        "review_body": "",
        "review_date": f"{label_id}",
        "labels": {
            f"{label_id}": {
                "createdAt": f'{datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}',
                "usageOptions": [],
                "scores": {},
                "datasets": {},
                "metadata": {},
                "augmentations": [],
            }
        },
    }


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
        description="Let an OpenAI model generate reviews from a prompt."
    )
    arg_parser.add_argument(
        "--prompt-ids",
        "-p",
        default=[
            "mf_did_not_work_v1",
            "mf_stopped_working_v1",
            "mf_sarcastic_v1",
            "mf_did_not_test_v1",
        ],
        nargs="*",
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
        default=20,
        type=int,
        help="Number of workers to use",
    )
    arg_parser.add_argument(
        "--save-percentage",
        "-s",
        choices=range(1, 101),
        type=int,
        default=100,
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
        default=1,
        help="Set the temperature for the sampling",
        metavar="[0.0-2.0]",
    )
    arg_parser.add_argument(
        "--num-reviews-per-prompt",
        "-n",
        default=[375, 250, 125, 250],
        nargs="*",
        type=int,
        help="Number of reviews to generate",
    )
    arg_parser.add_argument("file_path", help="json file path to save the reviews")

    return arg_parser.parse_args(), arg_parser.format_help()


async def generate_reviews(
    prompt_ids: str,
    label_id_suffix: str,
    workers: int,
    save_percentage: int,
    model: str,
    temperature: float,
    num_reviews_per_prompt: int,
    file_path: str,
):
    prompt_file_name = f"{BASE_PATH}/openai_api/prompts.json"
    prompts = []
    with open(prompt_file_name, "r") as prompt_file:
        prompts_json = json.load(prompt_file)
        prompt_type = "data-generation"
        for prompt_id in prompt_ids:
            if prompt_id not in prompts_json[prompt_type].keys():
                exit(
                    f"Prompt {prompt_id} of type {prompt_type} not found in {prompt_file_name}."
                )
            prompts.append(prompts_json[prompt_type][prompt_id]["prompt"])

    intermediate_save_size = 20
    count = 0
    reviews = {}

    def generate_label_worker_item(messages: str, review: dict):
        def task():
            nonlocal count
            reply = generate_review(
                messages=messages, model=model, temperature=temperature
            )["message"]["content"]
            print(f"Reply: {reply}")
            try:
                (
                    review["product_title"],
                    review["review_headline"],
                    review["review_body"],
                ) = reply.split("; ")[:3]
            except:
                try:
                    replies = reply.split("\n\n")
                    review["product_title"] = replies[0].split(": ")[1]
                    review["review_headline"] = replies[1].split(": ")[1]
                    review["review_body"] = replies[2].split(": ")[1]
                except:
                    try:
                        replies = reply.split("\n")
                        review["product_title"] = replies[0].split(": ")[1]
                        review["review_headline"] = replies[1].split(": ")[1]
                        review["review_body"] = replies[2].split(": ")[1]
                    except:
                        try:
                            replies = reply.split("; ")
                            review["product_title"] = replies[0].split(": ")[1]
                            review["review_headline"] = replies[1].split(": ")[1]
                            review["review_body"] = replies[2].split(": ")[1]
                        except:
                            (
                                review["product_title"],
                                review["review_headline"],
                                review["review_body"],
                            ) = ["Error", "Error", reply]

            reviews[
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ] = review

            if count % intermediate_save_size == 0 and count != 0:
                result = {
                    "version": REVIEWSET_VERSION,
                    "reviews": reviews,
                }
                with open(file_path, "w") as file:
                    json.dump(result, file)
                print(
                    f"{count}/{sum(num_reviews_per_prompt)} reviews processed. Intermediate results saved. 🚀"
                )

            count += 1

        return task

    labelling_queue = asyncio.Queue()

    for i, num_reviews in enumerate(num_reviews_per_prompt):
        label_id = "generate-{model_name}-{prompt_id}{hyphen}{id}".format(
            model_name=MODEL_NAME_MAPPING[model],
            prompt_id=prompt_ids[i],
            id=label_id_suffix,
            hyphen="-" if label_id_suffix != "" else "",
        )
        review = get_default_review(label_id)
        for _ in range(num_reviews):
            category = random.choice(PRODUCT_CATEGORIES)
            prompt = deepcopy(prompts[i])
            prompt[-1]["content"] += f"{category}"
            labelling_queue.put_nowait(
                generate_label_worker_item(prompt, review.copy())
            )

    worker = Worker(labelling_queue, n=workers)
    print(
        f"Setup queue with {worker.queue.qsize()} reviews. Starting {worker.n} workers in parallel 🚀"
    )

    await worker.run()

    print("All reviews processed. 🌪️🌪️🌪️")
    result = {
        "version": REVIEWSET_VERSION,
        "reviews": reviews,
    }
    with open(file_path, "w") as file:
        json.dump(result, file)


def generate_review(messages: str, model: str, temperature: float):
    return request_openai_api(
        lambda: openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=temperature,
        )
    )


def main():
    args, __ = parse_args()
    if len(args.prompt_ids) != len(args.num_reviews_per_prompt):
        exit(
            f"Number of prompt ids ({len(args.prompt_ids)}) and number of reviews per prompt ({len(args.num_reviews_per_prompt)}) do not match."
        )
    asyncio.run(generate_reviews(**vars(args)))


if __name__ == "__main__":
    main()
