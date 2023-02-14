import json
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
import evaluate

from utils.extract_reviews import extract_reviews_with_usage_options_from_json


# models for string similarity will only be loaded when needed
spacy_eval = bleu_eval = sacrebleu_eval = rouge_eval = st_eval = None


def extract_review_with_id(df: pd.DataFrame, review_id: str) -> Optional[pd.Series]:
    review = df[df.review_id == review_id]
    if review.empty:
        return None
    return review.iloc[0]


def human_predictions_to_labels(
    predictions_path: Union[Path, str],
    ground_truth_path: Union[Path, str],
    origin: Optional[str],
):
    vendor_data = extract_reviews_with_usage_options_from_json(predictions_path)
    golden_data = extract_reviews_with_usage_options_from_json(ground_truth_path)

    labels = []
    for _, predicted_review in predicted_df.iterrows():
        golden_review = extract_review_with_id(golden_df, predicted_review.review_id)
        if golden_review is not None:
            labels.append(
                {
                    "review_id": predicted_review.review_id,
                    "references": golden_review.usage_options,
                    "predictions": predicted_review.usage_options,
                    "origin": origin if origin else predicted_review.workerId,
                }
            )

    return labels


def gpt_predictions_to_labels(path: Union[Path, str]):
    labels = []
    with open(path) as json_file:
        data = json.load(json_file)
        for review in data["reviews"]:
            references = review["label"]["customUsageOptions"] + [
                " ".join(j["tokens"]) for j in review["label"]["annotations"]
            ]

            for prompt_id, predictions in review["label"].items():
                if prompt_id not in ["isFlagged", "annotations", "customUsageOptions"]:
                    labels.append(
                        {
                            "review_id": review["review_id"],
                            "references": references,
                            "predictions": predictions,
                            "origin": prompt_id,
                        }
                    )
    return labels


def get_similarity(
    prediction: str,
    reference: str,
    str_sim: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
) -> float:
    global st_eval, spacy_eval, bleu_eval, sacrebleu_eval, rouge_eval

    if use_lowercase:
        prediction = prediction.lower()
        reference = reference.lower()

    if str_sim == "all-mpnet-base-v2":
        if st_eval is None:
            st_eval = SentenceTransformer("all-mpnet-base-v2")
        embeddings1 = st_eval.encode(prediction, convert_to_tensor=True)
        embeddings2 = st_eval.encode(reference, convert_to_tensor=True)
        return util.cos_sim(embeddings1, embeddings2)[0][0].item()

    elif str_sim == "spacy":
        if spacy_eval is None:
            spacy_eval = spacy.load("en_core_web_md")
        prediction_tokens = spacy_eval(prediction)
        reference_tokens = spacy_eval(reference)
        return prediction_tokens.similarity(reference_tokens)

    elif str_sim == "bleu":
        if bleu_eval is None:
            bleu_eval = evaluate.load("bleu")
        pr, re = [prediction], [[reference]]
        return bleu_eval.compute(predictions=pr, references=re)["bleu"]

    elif str_sim == "sacrebleu":
        if sacrebleu_eval is None:
            sacrebleu_eval = evaluate.load("sacrebleu")
        res = sacrebleu_eval.compute(predictions=[prediction], references=[[reference]])
        return res["score"] * 0.01

    else:
        if rouge_eval is None:
            rouge_eval = evaluate.load("rouge")
        pr, re = [prediction], [[reference]]
        rogue_metrics = rouge_eval.compute(predictions=pr, references=re)
        # currently available: rouge1, rouge2, rougeL, rougeLsum
        if str_sim in rogue_metrics.keys():
            return rogue_metrics[str_sim]
        raise ValueError(f"metric {str_sim} is not supported")


def get_most_similar(
    label: str,
    options: list[str],
    str_sim: str = "all-mpnet-base-v2",
    threshold_word_sim: float = 0,
) -> tuple[float, str]:
    """For a single `label`, find the most similar match from `options`.

    Returns tuple (best similarity score, option with best similiarity score)."""
    assert 0 <= threshold_word_sim <= 1

    result = (0, None)
    for option in options:
        similarity = get_similarity(option, label, str_sim)
        if similarity >= max(result[0], threshold_word_sim):
            result = (similarity, option)

    return result
