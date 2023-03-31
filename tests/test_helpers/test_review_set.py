import copy
import json
import pytest
import os

from helpers.review_set import ReviewSet
from helpers.review import Review


@pytest.fixture
def review_set():
    return ReviewSet.from_files("tests/test_helpers/mock_review.json")


def test_len(review_set):
    assert len(review_set) == 2


def test_equals(review_set):
    review_set_copy = copy.deepcopy(review_set)
    assert review_set_copy == review_set


def test_save(review_set, tmp_path):
    with pytest.raises(AssertionError):
        review_set.save()  # ReviewSet class doesn't know where to save

    save_path = tmp_path / "mock_review_saved.json"

    review_set.save_as(save_path)
    loaded_review_set = ReviewSet.from_files(save_path)
    assert loaded_review_set == review_set

    os.remove(save_path)
    review_set.save()  # saving again should work because we called save_as before
    assert loaded_review_set == review_set


def test_drop_review_id(review_set):
    review_id_to_drop = "R1QANMA51Z124S"

    assert review_id_to_drop in review_set
    review_set.drop_review(review_id_to_drop)

    assert review_id_to_drop not in review_set


def test_drop_review(review_set):
    review_id_to_drop = "R1QANMA51Z124S"
    review_to_drop = review_set[review_id_to_drop]

    assert review_to_drop in review_set
    review_set.drop_review(review_to_drop)

    assert review_to_drop not in review_set


def almost_equal(a, b, threshold=0.000001):
    return abs(a - b) < threshold


def test_get_metrics(review_set):
    label_id = "openai-chat_gpt_4_matthisv1-cluster_test"
    reference_label_id = "bp-golden_v2"
    scores = review_set.get_agg_scores(label_id, reference_label_id)

    assert almost_equal(scores["custom_weighted_mean_recall"]["mean"], 0.5)
    assert almost_equal(scores["custom_weighted_mean_precision"]["mean"], 0.5)
    assert almost_equal(scores["custom_weighted_mean_f1"]["mean"], 0.5)
