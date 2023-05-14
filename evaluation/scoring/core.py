from pathlib import Path
from typing import Optional, Union

import evaluate
import pandas as pd

from evaluation.scoring.evaluation_cache import EvaluationCache
from helpers.extract_reviews import extract_reviews_with_usage_options_from_json
from openai_api.openai_backend import (
    DEFAULT_OPENAI_SIM_PARAMS,
    get_phrase_similiarity_from_openai,
)

# models for string similarity will only be loaded when needed
spacy_eval = bleu_eval = sacrebleu_eval = rouge_eval = st_eval = None


OPENAI_SIMILIARITY_CLASSES = {
    "identical": 1,
    "very similar": 0.9,
    "somewhat similar": 0.5,
    "dissimilar": 0,
}


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
    for _, predicted_review in vendor_data.iterrows():
        golden_review = extract_review_with_id(golden_data, predicted_review.review_id)
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


def get_embedding(usage_option: str, comparator: str = "all-mpnet-base-v2") -> list:
    global st_eval, spacy_eval, bleu_eval, sacrebleu_eval, rouge_eval

    cache = EvaluationCache.get()
    key = (comparator, usage_option)
    if key in cache:
        return cache[key]

    if comparator == "all-mpnet-base-v2":
        if st_eval is None:
            from sentence_transformers import SentenceTransformer

            st_eval = SentenceTransformer("all-mpnet-base-v2")
        embedding = st_eval.encode(usage_option)
    elif comparator == "spacy":
        if spacy_eval is None:
            import spacy

            spacy_eval = spacy.load("en_core_web_md")
        embedding = spacy_eval(usage_option)
    else:
        raise ValueError(f"embeddings for metric {comparator} doesn't exist")

    cache[key] = embedding
    return embedding

def get_all_similarities(predictions: str, reference: str, use_lowercase: bool=True, openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS):
    for comparator in ["all-mpnet-base-v2", "bleu", "sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "openai"]:
        yield comparator, get_similarity(predictions, reference, comparator, use_lowercase, openai_params)


def get_similarity(
    prediction: str,
    reference: str,
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
) -> float:
    global st_eval, spacy_eval, bleu_eval, sacrebleu_eval, rouge_eval

    if use_lowercase:
        prediction = prediction.lower()
        reference = reference.lower()

    if modification == "stem":
        import nltk

        nltk.download("punkt", quiet=True)
        ps = nltk.stem.PorterStemmer()
        prediction = " ".join(ps.stem(word) for word in prediction.split())
        reference = " ".join(ps.stem(word) for word in reference.split())
    elif modification == "lemmatize":
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        wnl = nltk.stem.WordNetLemmatizer()
        prediction = " ".join(wnl.lemmatize(word) for word in prediction.split())
        reference = " ".join(wnl.lemmatize(word) for word in reference.split())

    if comparator == "openai":
        cache = EvaluationCache.get()
        key = tuple(
            [comparator]
            + list(sorted([prediction, reference]))
            + [
                openai_params["model"],
                openai_params["prompt_id"],
                openai_params["temperature"],
            ]
        )

        if key in cache:
            return cache[key]

        similiarity_class = get_phrase_similiarity_from_openai(
            prediction, reference, **openai_params
        )
        if similiarity_class in OPENAI_SIMILIARITY_CLASSES:
            similarity = OPENAI_SIMILIARITY_CLASSES[similiarity_class]
        else:
            similarity = 0
            print(
                f"WARNING: '{similiarity_class}' is not a valid similarity class for prediction '{prediction}' and reference '{reference}'"
            )

        cache[key] = similarity
        return similarity
    elif comparator == "all-mpnet-base-v2" or comparator == "spacy":
        prediction_tokens = get_embedding(prediction, comparator)
        reference_tokens = get_embedding(reference, comparator)
        if comparator == "all-mpnet-base-v2":
            from sentence_transformers import util

            similarity = util.cos_sim(prediction_tokens, reference_tokens)[0][0].item()
        else:
            similarity = prediction_tokens.similarity(reference_tokens)
        return similarity
    elif comparator == "bleu":
        if bleu_eval is None:
            bleu_eval = evaluate.load("bleu")
        pr, re = [prediction], [[reference]]
        return bleu_eval.compute(predictions=pr, references=re)["bleu"]

    elif comparator == "sacrebleu":
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
        if comparator in rogue_metrics.keys():
            return rogue_metrics[comparator]
        else:
            raise ValueError(f"comparator {comparator} is not supported")


def get_most_similar(
    label: str,
    options: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    openai_params: dict = DEFAULT_OPENAI_SIM_PARAMS,  # only needed for comparator "openai"
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    threshold_word_sim: float = 0,
) -> tuple[float, str]:
    """For a single `label`, find the most similar match from `options`.

    Returns tuple (best similarity score, option with best similiarity score)."""
    assert 0 <= threshold_word_sim <= 1

    result = (0, None)
    for option in options:
        similarity = get_similarity(
            prediction=option,
            reference=label,
            comparator=comparator,
            use_lowercase=use_lowercase,
            openai_params=openai_params,
            modification=modification,
        )
        if similarity >= max(result[0], threshold_word_sim):
            result = (similarity, option)

    return result
