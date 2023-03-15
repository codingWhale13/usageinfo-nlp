import os
import time
import json
import openai

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID", "org-wud6DQs34D79lUPQBuJnHo4f")
no_usage_option_string = "No use cases"
chat_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]
model_name_mapping = {
    "text-davinci-003": "davinci",
    "gpt-3.5-turbo": "chat_gpt",
    "gpt-3.5-turbo-0301": "chat_gpt",
}

OPENAI_MAX_RETRIES = 10


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


def get_chat_labels_from_openai(
    review: json, messages: dict(), model: str, temperature: float
):
    def helper(message, review):
        message = message.copy()
        message["content"] = eval('f"""' + message["content"] + '"""')
        return message

    messages = [helper(message, review) for message in messages]

    openai.organization = openai_org_id
    openai.api_key = openai_api_key
    api_failure_count = 0

    while api_failure_count < OPENAI_MAX_RETRIES:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=temperature,
                timeout=8,
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


def format_usage_options(text_completion: str):
    labels = []
    for label in text_completion.split(","):
        if label.strip().startswith(no_usage_option_string):
            break
        labels.append(label.strip().strip("."))
    return labels


def generate_label(
    review: json,
    prompt: str | list,
    model: str = "text-davinci-003",
    temperature: float = 1,
    logprobs: int = None,
    prompt_id: str = None,
):
    metaData = {
        "openai": {"model": model, "temperature": temperature, "prompt_id": prompt_id}
    }
    if model in chat_models:
        output = get_chat_labels_from_openai(review, prompt, model, temperature)
        usage_options = format_usage_options(output.message["content"].strip())
    else:
        output = get_labels_from_openai(review, prompt, model, temperature, logprobs)
        usage_options = format_usage_options(output.text.strip())
        usage_options_logprobs = aggregate_logprobs(output)
        metaData["openai"].update(
            {
                "logprobs": output.logprobs,
                "usageOptions_logporbs": usage_options_logprobs,
            }
        )

    return usage_options, metaData


def add_label(review: json, label_id: str, label: json):
    if "labels" not in review.keys():
        review["labels"] = {}

    review["labels"][label_id] = label
