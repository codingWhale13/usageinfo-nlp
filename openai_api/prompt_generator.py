#!/usr/bin/env python3
import argparse
import random
import json
import os
from copy import copy
from helpers.review_set import ReviewSet


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Amend some review examples from a json file to a prompt template."
    )
    arg_parser.add_argument(
        "base_prompt",
        type=str,
        help="Name of the prompt that you want to amend examples to",
    )
    arg_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the reviewset that you want to take examples from",
    )
    arg_parser.add_argument(
        "label_id",
        type=str,
        help="label id that you want to use",
    )
    arg_parser.add_argument(
        "n",
        type=int,
        help="Number of review examples to amend to the prompt",
    )
    arg_parser.add_argument(
        "new_prompt",
        type=str,
        help="Name of the new prompt with the amended examples",
    )
    arg_parser.add_argument(
        "--order",
        "-o",
        action="store_true",
        help="Take the first n examples from the reviewset in order (by default, the reviewset is shuffled first)",
    )
    arg_parser.add_argument(
        "--system",
        "-s",
        action="store_true",
        help="Use if detailed instructions are part of the system message",
    )
    arg_parser.add_argument(
        "--first",
        "-f",
        action="store_true",
        help="Use if there is no example in the template prompt yet",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def main():
    args, _ = parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    prompt_file_name = f"{script_dir}/prompts.json"
    with open(prompt_file_name, "r") as prompt_file:
        prompts_json = json.load(prompt_file)
    if args.base_prompt not in prompts_json["chat"].keys():
        exit(f"Prompt {args.base_prompt} not found in {prompt_file_name}.")
    prompt = copy(prompts_json["chat"][args.base_prompt]["prompt"])

    review_set = ReviewSet.from_files(args.base_file)
    if not args.label_id in review_set.get_all_label_ids():
        exit(f"Label id {args.label_id} not found in {args.base_file}.")

    if not args.order:
        reviews = sorted(list(review_set), key=lambda review: review.review_id)
        random.shuffle(reviews)
        review_set = ReviewSet.from_reviews(*reviews)

    sub_reviewset = review_set[: args.n]

    for count, review in enumerate(sub_reviewset):
        if args.system:
            question = {
                "role": "user",
                "content": f"{review['review_body']}",
            }
        else:
            if count == 0 and args.first:
                question = prompt.pop()
                question[
                    "content"
                ] = f"{question['content']}\nHere is a review:\n{review['review_body']}"
            else:
                question = {
                    "role": "user",
                    "content": f"Here is another review:\n{review['review_body']}",
                }

        usage_options = "; ".join(review.get_usage_options(args.label_id))
        answer = {
            "role": "assistant",
            "content": f"{usage_options or 'No usage options'}",
        }
        prompt.append(question)
        prompt.append(answer)

    if args.system:
        final_question = {"role": "user", "content": "{review['review_body']}"}
        prompt.append(final_question)
    else:
        final_question = {
            "role": "user",
            "content": "Here is another review:\n{review['review_body']}",
        }
        prompt.append(final_question)

    prompts_json["chat"][f"{args.new_prompt}"] = {"prompt": prompt}
    with open(prompt_file_name, "w") as prompt_file:
        json.dump(prompts_json, prompt_file)


if __name__ == "__main__":
    main()
