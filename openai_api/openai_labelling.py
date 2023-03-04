import os
import time
import argparse
import json
import openai


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID", "org-wud6DQs34D79lUPQBuJnHo4f")
no_usage_option_string = "No use cases"

OPENAI_MAX_RETRIES = 10


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
        "--prompt", "-p", default="nils_v2", help="Prompt to use for openai labelling"
    )
    arg_parser.add_argument(
        "--id", default="", help="Unique identifier for the new label"
    )
    arg_parser.add_argument(
        "--save",
        "-s",
        choices=range(1, 101),
        type=int,
        default=5,
        help="Make an intermediate save every n percent of processed reviews",
        metavar="[0-100]",
    )
    arg_parser.add_argument(
        "--model", "-m", default="text-davinci-003", help="OpenAI model to use"
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
        "--logprobs",
        "-l",
        choices=range(1, 6),
        type=int,
        default=5,
        help="Define how many logprobs you want to save",
        metavar="[1-5]",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def get_labels_from_openai(
    review: json, prompt: str, model: str, temperature: float, logprobs: int
):
    prompt_with_review = eval('f"""' + prompt + '"""')

    openai.organization = openai_org_id
    openai.api_key = openai_api_key
    api_failure_count = 0

    while api_failure_count < OPENAI_MAX_RETRIES:
        try:
            completion = openai.Completion.create(
                model=model,
                prompt=prompt_with_review,
                max_tokens=500,
                logprobs=logprobs,
                temperature=temperature,
            )
            api_failure_count = 0
            return completion.choices[0]
        except openai.OpenAIError as openai_error:
            api_failure_count += 1
            wait_time = 2**api_failure_count
            print("WARNING: OpenAI API error: " + str(openai_error))
            print(f"Waiting {wait_time} seconds and trying again...")
            time.sleep(wait_time)
    raise Exception(
        f"Max Oopenai retry counter of {OPENAI_MAX_RETRIES} exceeded with {api_failure_count} retires"
    )


def aggregate_logprobs(output: json):
    logprobs = []
    current_usage_option_logprob = 0

    for token, logprob in zip(
        output["logprobs"]["tokens"], output["logprobs"]["token_logprobs"]
    ):
        if token == "\n":
            continue
        elif token.strip() == ",":
            logprobs.append(current_usage_option_logprob)
            current_usage_option_logprob = 0
        else:
            current_usage_option_logprob += logprob
    logprobs.append(current_usage_option_logprob)

    return logprobs


def format_usageOptions(text_completion: str):
    labels = []
    for label in text_completion.split(","):
        if label.strip().startswith(no_usage_option_string):
            break
        labels.append(label.strip().strip("."))
    return labels


def generate_label(
    review: json,
    prompt: str,
    model: str = "text-davinci-003",
    temperature: float = 1,
    logprobs: int = None,
):
    output = get_labels_from_openai(review, prompt, model, temperature, logprobs)

    usageOptions = format_usageOptions(output.text.strip())
    usageOptions_logprobs = aggregate_logprobs(output)
    metaData = {
        "openai": {
            "model": model,
            "temperature": temperature,
            "usageOptions_logporbs": usageOptions_logprobs,
            "logprobs": output.logprobs,
        }
    }

    return {"usageOptions": usageOptions, "metaData": metaData}


def add_label(review: json, label_id: str, label: json):
    if "labels" not in review.keys():
        review["labels"] = {}

    review["labels"][label_id] = label


def get_all_label_ids(review_json: json):
    label_ids = set()
    for review in review_json["reviews"]:
        if "labels" in review.keys():
            label_ids.update(review["labels"].keys())
    return label_ids


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    args, help_text = parse_args()

    review_file_name = os.path.abspath(args.file)
    prompt_file_name = f"{script_dir}/prompts.json"

    prompt_name = args.prompt
    with open(prompt_file_name, "r") as prompt_file:
        prompts_json = json.load(prompt_file)
        if prompt_name not in prompts_json.keys():
            exit(f"Prompt {prompt_name} not found in {prompt_file_name}.")
        prompt = prompts_json[prompt_name]["prompt"]

    label_id = "davinci-{prompt_name}{hyphen}{id}".format(
        prompt_name=prompt_name, id=args.id, hyphen="-" if args.id != "" else ""
    )

    with open(review_file_name, "r+") as review_file:
        review_json = json.load(review_file)

        if label_id in get_all_label_ids(review_json):
            if (
                input(f"Label {label_id} already exists. Overwrite? (y/N): ").lower()
                != "y"
            ):
                exit("Aborted.")

        num_reviews = len(review_json["reviews"])
        intermediate_save_size = max(1, int(num_reviews * (args.save / 100)))

        for count, review in enumerate(review_json["reviews"]):
            add_label(
                review,
                label_id,
                generate_label(
                    review,
                    prompt,
                    model=args.model,
                    temperature=args.temperature,
                    logprobs=args.logprobs,
                ),
            )

            if count % intermediate_save_size == 0 and count != 0:
                review_file.truncate(0)
                review_file.seek(0)
                json.dump(review_json, review_file, indent=4)
                print(
                    f"{count} of {num_reviews} reviews processed. Intermediate results saved."
                )

        review_file.truncate(0)
        review_file.seek(0)
        json.dump(review_json, review_file, indent=4)
        print(f"All reviews processed.")


if __name__ == "__main__":
    main()
