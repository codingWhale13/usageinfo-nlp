from src.openai_api.openai_pre_annotion import pre_label_with_logprobs
import json
from collections import Counter
import math
from statistics import mean

FIRST_STEP_SAMPLES = 2
SECOND_STEP_SAMPLES = 2
MAX_SAMPLES = 5


def select_most_likely_sample(samples, agg=mean, desired_output_type=None, review=None):
    most_likeley_sample = None
    best_logprob = -math.inf

    for sample in samples:
        if desired_output_type is None or output_type(sample) == desired_output_type:
            logprob = agg(sample["usageOptionsLogprobs"])
            if logprob > best_logprob:
                best_logprob = logprob
                most_likeley_sample = sample
    print(f"Selected output type: {output_type(most_likeley_sample)}")
    if review is not None:
        with open(f"openai_samples/{review['review_id']}.json", "w") as json_file:
            json.dump(samples, json_file)
    return most_likeley_sample


def generate_samples(
    review: json, model: str = "text-davinci-003", prompt: str = None, n: int = 1
):
    return [
        pre_label_with_logprobs(review, model=model, prompt=prompt) for _ in range(n)
    ]


def output_type(sample):
    return True if sample["usageOptions"][0] != None else False


def count_output_types(samples):
    output_types = [output_type(output) for output in samples]
    return Counter(output_types)


def pre_label_with_dynamic_samplling(
    review: json, model: str = "text-davinci-003", prompt: str = None
):
    samples = generate_samples(review, model=model, prompt=prompt, n=FIRST_STEP_SAMPLES)
    print(f"Sampled {FIRST_STEP_SAMPLES} labels")
    output_types_counter = count_output_types(samples)
    if len(output_types_counter.keys()) == 1:
        print(f"Labels agree to be the same output type: {output_types_counter.keys()}")
        return select_most_likely_sample(samples, review=review)
    else:
        samples = samples + generate_samples(
            review, model=model, prompt=prompt, n=SECOND_STEP_SAMPLES
        )
        output_types_counter = count_output_types(samples)
        if output_types_counter[True] >= FIRST_STEP_SAMPLES + SECOND_STEP_SAMPLES - 1:
            print(f"3/4 for true")
            return select_most_likely_sample(
                samples, desired_output_type=True, review=review
            )
        elif (
            output_types_counter[False] >= FIRST_STEP_SAMPLES + SECOND_STEP_SAMPLES - 1
        ):
            print(f"3/4 for false")
            return select_most_likely_sample(
                samples, desired_output_type=False, review=review
            )
        else:
            samples = samples + generate_samples(
                review, model=model, prompt=prompt, n=1
            )
            if output_types_counter[True] > output_types_counter[False]:
                print("Majority for true")
                return select_most_likely_sample(
                    samples, desired_output_type=True, review=review
                )
            else:
                print("Majority for false")
                return select_most_likely_sample(
                    samples, desired_output_type=False, review=review
                )
