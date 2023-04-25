import os
import time
import json
import openai
from liquid import Template
from openai_api.parsing import format_usage_options
import dotenv

dotenv_path = dotenv.find_dotenv()
is_dotenv_found = dotenv.load_dotenv(dotenv_path)
if is_dotenv_found is False:
    print("Warning no .env file found")
else:
    print(f"Sucessfully loaded {dotenv_path} as dotenv file")

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID", "org-wud6DQs34D79lUPQBuJnHo4f")
chat_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314"]
model_name_mapping = {
    "text-davinci-003": "davinci",
    "gpt-3.5-turbo": "chat_gpt",
    "gpt-3.5-turbo-0301": "chat_gpt",
    "gpt-4": "gpt_4",
    "gpt-4-0314": "gpt_4",
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
        template = Template(message["content"])
        message["content"] = template.render(review_body=review["review_body"])
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


def generate_label(
    review: json,
    prompt: str | list,
    model: str = "text-davinci-003",
    temperature: float = 1,
    logprobs: int = None,
    prompt_id: str = None,
):
    metadata = {"model": model, "temperature": temperature, "prompt_id": prompt_id}
    if model in chat_models:
        raw_output = (
            get_chat_labels_from_openai(review, prompt, model, temperature)
            .message["content"]
            .strip()
        )
        usage_options, extracted_metadata = format_usage_options(raw_output)
        print(extracted_metadata, usage_options)
        metadata.update({"raw_output": raw_output} | extracted_metadata)
    else:
        output = get_labels_from_openai(review, prompt, model, temperature, logprobs)
        raw_output = output.text.strip()
        usage_options, extracted_metadata = format_usage_options(raw_output)
        usage_options_logprobs = aggregate_logprobs(output)
        metadata.update(
            {
                "logprobs": output.logprobs,
                "usageOptions_logporbs": usage_options_logprobs,
                "raw_output": raw_output,
            }
            | extracted_metadata
        )

    return usage_options, {"openai": metadata}
