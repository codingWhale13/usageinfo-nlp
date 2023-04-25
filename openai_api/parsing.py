NO_USAGE_OPTION_STRING = "No use cases"


def extract_text_after_keyword(
    text: str,
    keyword: str,
    stop_words=list[str],
):
    if keyword not in text:
        raise ValueError("Keyword not in text")
    releveant_substring = text[text.index(keyword) + len(keyword) :]
    stop_word_indexes = [len(releveant_substring) + 1] + [
        releveant_substring.index(stop_word)
        for stop_word in stop_words
        if stop_word in releveant_substring
    ]
    return releveant_substring[: min(stop_word_indexes)]


def text_completion_is_in_explanation_v1_format(text_completion: str):
    return all(
        keyword in text_completion
        for keyword in ["Explanation:", "Result:", "Sentiment:", "Confidence:"]
    )


def extract_text_after_keyword_from_explanation_v1_format(
    text_completion: str, keyword: str
):
    return extract_text_after_keyword(
        text_completion,
        keyword,
        stop_words=["Result:", "Explanation:", "Sentiment:", "Confidence:"],
    )


def format_usage_options(text_completion: str):
    metadata = extract_metadata(text_completion)
    if text_completion_is_in_explanation_v1_format(text_completion):
        text_completion = extract_text_after_keyword_from_explanation_v1_format(
            text_completion, "Result:"
        ).strip()
    labels = []
    for label in text_completion.split(","):
        if label.strip().startswith(NO_USAGE_OPTION_STRING):
            break
        labels.append(label.strip().strip("."))
    return labels, metadata


def percentage_to_float(s: str):
    return float(s.strip().strip("%")) / 100


def extract_metadata(text_completion: str) -> dict[str]:
    metadata = {}
    if text_completion_is_in_explanation_v1_format(text_completion):
        confidence_string = extract_text_after_keyword_from_explanation_v1_format(
            text_completion, "Confidence:"
        ).strip()
        try:
            metadata["confidence"] = percentage_to_float(confidence_string)
        except:
            metadata["confidence"] = confidence_string
        metadata["explanation"] = extract_text_after_keyword_from_explanation_v1_format(
            text_completion, "Explanation:"
        ).strip()
        metadata["sentiment"] = extract_text_after_keyword_from_explanation_v1_format(
            text_completion, "Sentiment:"
        ).strip()
    return metadata
