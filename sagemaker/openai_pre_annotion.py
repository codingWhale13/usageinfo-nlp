import os
import argparse
import json
import openai


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID", "org-wud6DQs34D79lUPQBuJnHo4f")
no_usage_option_string = "No use cases"


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Let an OpenAI model pre-label the reviews in a json file."
    )
    arg_parser.add_argument("--file", "-f", required=True, help="json file path")

    return arg_parser.parse_args(), arg_parser.format_help()


def pre_label(review: str, model: str = "text-davinci-003"):
    prompt_with_review = f"""This is a review for an E-Commerce product by a customer:

{review}

Summarize what the product can be used for in real life according to the customer. Write one usage option or a list of only the most important use cases described with as few words as possible as a comma separated list. Don't consider negative use cases, gifts, presents, characteristics, attributes or product qualities. Write '"No use cases" if there are no valid use cases left:"""

    openai.organization = openai_org_id
    openai.api_key = openai_api_key

    completion = openai.Completion.create(
        engine=model,
        prompt=prompt_with_review,
        max_tokens=2048,
    )

    return completion.choices[0].text.strip()


def pre_label_format_manifest(review: str, model: str = "text-davinci-003"):
    output = pre_label(review, model)
    labels = []
    for label in output.split(","):
        if label.strip().startswith(no_usage_option_string):
            break

        labels.append(label.strip().strip("."))

    return labels


def main():
    args, help_text = parse_args()

    file_name = os.path.abspath(args.file)

    with open(file_name, "r") as file:
        review_json = json.load(file)

        for review in review_json["reviews"]:
            review["label"]["customUsageOptions"] = pre_label_format_manifest(
                review["review_body"]
            )

    output_file_name = (
        os.path.dirname(file_name) + "/pre_labelled_" + os.path.basename(file_name)
    )
    with open(output_file_name, "w") as file:
        json.dump(review_json, file)


if __name__ == "__main__":
    main()
