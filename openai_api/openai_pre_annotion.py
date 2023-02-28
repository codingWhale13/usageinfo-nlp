import os
import time
import argparse
import json
import openai


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID", "org-wud6DQs34D79lUPQBuJnHo4f")
no_usage_option_string = "No use cases"

api_failure_count = 0
MAX_RETRIES = 10


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Let an OpenAI model pre-label the reviews in a json file."
    )
    arg_parser.add_argument("--file", "-f", required=True, help="json file path")
    arg_parser.add_argument("--save", "-s", choices=range(1,101), type=int, default=5, help="Make an intermediate save every n percent of processed reviews", metavar="[0-100]")

    return arg_parser.parse_args(), arg_parser.format_help()


def pre_label(review: json, model: str = "text-davinci-003", prompt: str = None, logprobs: int=None):
    global api_failure_count
    if prompt is not None:
        prompt_with_review = eval('f"""' + prompt + '"""')
    else:
        prompt_with_review = f"""This is a review for an E-Commerce product by a customer.

{review['review_body']}

Summarize the customer's most important use cases for the product in real life, as a list of short phrases separated by commas. Only include positive use cases that the customer describes, and exclude mentions of gifts, presents, characteristics, attributes, or product qualities. If no valid use cases are mentioned in the review, write "No use cases.\""""

    openai.organization = openai_org_id
    openai.api_key = openai_api_key
    while api_failure_count < MAX_RETRIES:
        try:
            completion = openai.Completion.create(
                engine=model,
                prompt=prompt_with_review,
                max_tokens=200,
                logprobs=logprobs,
                temperature=0.7
            )
            api_failure_count = 0
            text_completion = completion.choices[0].text.strip()
            if logprobs == None:
                return text_completion
            else:
                return {"text": text_completion, "logprobs": completion.choices[0].logprobs}
        except openai.OpenAIError as e:
            api_failure_count += 1
            wait_time = 2**api_failure_count
            print("WARNING: OpenAI API error: " + str(e))
            print(f"Waiting {wait_time} seconds and trying again...")
            time.sleep(wait_time)
    if api_failure_count >= MAX_RETRIES:
        raise Exception(
            f"Max openai retry counter of {MAX_RETRIES} exceeded with {api_failure_count} retires"
        )


def pre_label_format_manifest(
    review: json, model: str = "text-davinci-003", prompt: str = None, logprobs: int=None
):
    output = pre_label(review, model, prompt, logprobs=logprobs)
    if type(output) is str:
        text_completion = output
    else:
        text_completion = output['text']
    labels = []
    for label in text_completion.split(","):
        if label.strip().startswith(no_usage_option_string):
            break

        labels.append(label.strip().strip("."))
    if logprobs is None:
        return labels
    else:
        return {'usageOptions': labels, "logprobs": output['logprobs']}

def pre_label_with_logprobs(
    review: json, model: str = "text-davinci-003", prompt: str = None, logprobs: int=5
):
   
    
    output = pre_label(review, model, prompt, logprobs=logprobs)
    labels = []
    logprobs = []
    current_usage_option = ""
    current_usage_option_logprob = 0

    def add_new_usage_option(current_usage_option, current_usage_option_logprob):
        if current_usage_option.strip().startswith(no_usage_option_string):
            labels.append(None)
            logprobs.append(current_usage_option_logprob)
        else:
            labels.append(current_usage_option.strip().replace('.',''))
            logprobs.append(current_usage_option_logprob)

        current_usage_option = ""
        current_usage_option_logprob = 0
    for token, logprob in zip(output['logprobs']['tokens'], output['logprobs']['token_logprobs']):
        if token == '\n':
            continue
        elif token.strip() == ',':
            add_new_usage_option(current_usage_option, current_usage_option_logprob)  
        else:
            current_usage_option += token
            current_usage_option_logprob += logprob
    add_new_usage_option(current_usage_option, current_usage_option_logprob)

    
    return {'usageOptions': labels, "usageOptionsLogprobs": logprobs, "logprobs": output['logprobs']}
    
def main():
    args, help_text = parse_args()

    file_name = os.path.abspath(args.file)
    output_file_name = (
        os.path.dirname(file_name) + "/pre_labelled_" + os.path.basename(file_name)
    )

    prompt = "I will give you a customer review for an e-commerce product. You should answer the question \"What can this product be used as / for\"? by only using information from the review author. In case the review author mentions multiple use cases, print them comma-separated. If the review author does not mention any use case, output \"No use cases\". Do not output negative use cases or further product information like product quality, attributes, target audiences, etc. Here is the review:\n{review['review_body']}\n"

    with open(file_name, "r") as file:
        review_json = json.load(file)
        if type(review_json) is dict:
            review_json = review_json["reviews"]
        num_reviews = len(review_json)

        intermediate_save_size = max(1, int(num_reviews * (args.save / 100)))

        labelled_review_json = []

        for count, review in enumerate(review_json):
            review["label"] = {}
            review["label"]["customUsageOptions"] = pre_label_format_manifest(review, prompt=prompt)
            labelled_review_json.append(review)

            if count % intermediate_save_size == 0 and count != 0:
                with open(output_file_name, "w") as file:
                    json.dump(labelled_review_json, file)
                print(f"{count} of {num_reviews} reviews processed. Intermediate results saved.")

        with open(output_file_name, "w") as file:
            json.dump(labelled_review_json, file)
        print(f"All reviews processed.")

if __name__ == "__main__":
    main()
