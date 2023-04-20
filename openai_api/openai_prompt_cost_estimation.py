#!/usr/bin/env python3
import os
import argparse
import json

DEFAULT_MODEL = "text-davinci-003"

WORDS_TO_TOKENS = 1000 / 750

PRICES_PER_1000_TOKENS = {"text-davinci-003": 0.02}

MEAN_WORDS_PER_REVIEW = 50
MEAN_WORDS_PER_OPENAI_USAGE_OPTIONS_OUTPUT = 15


def estimate_price_per_review(prompt, model):
    price = 0
    prompt_word_count = len(prompt["prompt"].split(" "))
    total_word_count = (
        prompt_word_count
        + MEAN_WORDS_PER_REVIEW
        + MEAN_WORDS_PER_OPENAI_USAGE_OPTIONS_OUTPUT
    )
    total_token_count = total_word_count * WORDS_TO_TOKENS
    return PRICES_PER_1000_TOKENS[model] * (total_token_count / 1000)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Estimate price per review for prompts"
    )
    arg_parser.add_argument(
        "--prompts", "-p", default="prompts.json", help="json file path with prompts"
    )

    arg_parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, help="OpenAI model. Influences price"
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def main():
    args, help_text = parse_args()

    prompts_file_name = os.path.abspath(args.prompts)
    openai_model_used = args.model
    with open(prompts_file_name, "r+") as prompts_file:
        prompts_json = json.load(prompts_file)

        for prompt in prompts_json:
            estimated_price_per_review = estimate_price_per_review(
                prompt, openai_model_used
            )
            print(f"{prompt['name']}: ${estimated_price_per_review}")


if __name__ == "__main__":
    main()
