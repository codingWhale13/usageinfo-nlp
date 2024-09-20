import json
import os
import time
from typing import Callable

import dotenv

dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID")

OPENAI_MAX_RETRIES = 10
NO_USAGE_OPTION_STR = "No usage options"
CHAT_MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314"]
MODEL_NAME_MAPPING = {
    "text-davinci-003": "davinci",
    "gpt-3.5-turbo": "chat_gpt",
    "gpt-3.5-turbo-0301": "chat_gpt",
    "gpt-4": "gpt_4",
    "gpt-4-0314": "gpt_4",
}
DEFAULT_OPENAI_SIM_PARAMS = {  # for phrase similarity
    "model": "gpt-3.5-turbo",
    "prompt_id": "nils_v1",
    "temperature": 1.0,
}


def request_openai_api(openai_function: Callable):
    import openai

    openai.organization = openai_org_id
    openai.api_key = openai_api_key
    api_failure_count = 0

    while api_failure_count < OPENAI_MAX_RETRIES:
        try:
            completion = openai_function()
            api_failure_count = 0
            return completion.choices[0]
        except openai.OpenAIError as openai_error:
            api_failure_count += 1
            wait_time = 2**api_failure_count
            print("WARNING: OpenAI API error: " + str(openai_error))
            print(f"Waiting {wait_time} seconds and trying again...")
            time.sleep(wait_time)
    raise Exception(f"too many requests to OpenAI failed {api_failure_count}")


def get_labels_from_openai(
    review: json, prompt: str, model: str, temperature: float, logprobs: int
):
    """NOTE: the `review` argument actually is used here, within the evaluated prompt!"""
    import openai

    return request_openai_api(
        lambda: openai.Completion.create(
            model=model,
            prompt=eval('f"""' + prompt + '"""'),
            max_tokens=500,
            logprobs=logprobs,
            temperature=temperature,
        )
    )


def chat_completion(model, messages, temperature):
    import openai

    return request_openai_api(
        lambda: openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
            timeout=8,
        )
    )


def get_chat_labels_from_openai(
    review: json, messages: dict(), model: str, temperature: float
):
    def helper(message, review):
        message = message.copy()
        message["content"] = eval('f"""' + message["content"] + '"""')
        return message

    messages = [helper(message, review) for message in messages]

    return chat_completion(model, messages, temperature)


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
    metadata = {"text_completion": text_completion}
    if "Result:" in text_completion:
        text_completion = text_completion.split("Result:", maxsplit=1)[1]

    for label in text_completion.split(";"):
        if label.strip().startswith(NO_USAGE_OPTION_STR):
            break
        labels.append(label.strip().strip("."))
    return labels, metadata


def generate_label(
    review: json,
    prompt: str | list,
    model: str = "text-davinci-003",
    temperature: float = 1,
    logprobs: int = None,
    prompt_id: str = None,
):
    metadata = {
        "openai": {"model": model, "temperature": temperature, "prompt_id": prompt_id}
    }
    if model in CHAT_MODELS:
        output = get_chat_labels_from_openai(review, prompt, model, temperature)
        usage_options, answer_metadata = format_usage_options(
            output.message["content"].strip()
        )
        metadata |= answer_metadata
    else:
        output = get_labels_from_openai(review, prompt, model, temperature, logprobs)
        usage_options, answer_metadata = format_usage_options(output.text.strip())
        usage_options_logprobs = aggregate_logprobs(output)
        metadata["openai"].update(
            {
                "logprobs": output.logprobs,
                "usageOptions_logporbs": usage_options_logprobs,
            }
            | answer_metadata
        )

    return usage_options, metadata


def get_phrase_similiarity_from_openai(
    phrase_1: str,
    phrase_2: str,
    model: str = "gpt-3.5-turbo",
    prompt_id: str = "nils_v1",
    temperature: float = 1.0,
):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, "prompts.json")) as prompt_file:
        messages = json.load(prompt_file)["phrase-similarity"][prompt_id]["prompt"]
        for message in messages:
            message["content"] = (
                message["content"]
                .replace("####1####", phrase_1)
                .replace("####2####", phrase_2)
            )

    # returns "dissimilar", "somewhat similar", "very similar", or "identical"
    return chat_completion(model, messages, temperature).message["content"].strip()
