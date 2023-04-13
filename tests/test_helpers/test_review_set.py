from copy import copy, deepcopy
import pytest
import os

from helpers.review_set import ReviewSet
from helpers.review import Review


@pytest.fixture
def review_set():
    return ReviewSet.from_files("tests/test_helpers/mock_review.json")


def test_equals(review_set):
    review_set_1 = review_set
    review_set_2 = deepcopy(review_set_1)

    assert review_set_1 == review_set_2

    review_set_1.drop_review("K1QFNMA51Z114S", inplace=True)
    assert review_set_1 != review_set_2


def test_or(review_set):
    review_set_1 = review_set
    review_set_2 = deepcopy(review_set_1)
    review_1, review_2 = deepcopy(review_set_1.reviews)

    review_set_1.drop_review(review_1, inplace=True)
    review_set_2.drop_review(review_2, inplace=True)

    assert len(review_set_1) == len(review_set_2) == 1

    merged_review_sets = review_set_1 | review_set_2
    assert (
        len(merged_review_sets) == 2
        and review_1 in merged_review_sets
        and review_2 in merged_review_sets
    )
    assert len(review_set_1) == len(review_set_2) == 1  # no side effects should happen


def test_ior(review_set):
    review_set_1 = review_set
    review_set_2 = deepcopy(review_set_1)
    review_1, review_2 = deepcopy(review_set_1.reviews)

    review_set_1.drop_review(review_1, inplace=True)
    review_set_2.drop_review(review_2, inplace=True)

    assert len(review_set_1) == len(review_set_2) == 1

    merged_review_sets = review_set_1
    merged_review_sets |= review_set_2
    assert (
        len(merged_review_sets) == 2
        and review_1 in merged_review_sets
        and review_2 in merged_review_sets
    )
    assert len(review_set_1) == 2
    assert len(review_set_2) == 1  # the merge should have no effect on review_set_2


def test_copy(review_set):
    review_set_1 = review_set
    review_set_2 = copy(review_set_1)

    assert id(review_set_1) != id(review_set_2)  # duh
    assert id(review_set_1.reviews) != id(review_set_2.reviews)

    review_id = "R1QANMA51Z124S"
    # the review objects didn't change - pretty shallow...
    assert id(review_set_1[review_id]) == id(review_set_2[review_id])


def test_deepcopy(review_set):
    review_set_1 = review_set
    review_set_2 = deepcopy(review_set_1)

    assert id(review_set_1) != id(review_set_2)  # duh
    assert id(review_set_1.reviews) != id(review_set_2.reviews)

    review_id = "R1QANMA51Z124S"
    # even the review objects changed - wow, that's deep!
    assert id(review_set_1[review_id]) != id(review_set_2[review_id])


def test_len(review_set):
    assert len(review_set) == 2


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
    label_id = "openai-chat_gpt_4_matthisv1-test"
    reference_label_id = "bp-golden_v2"
    scores = review_set.get_agg_scores(label_id, reference_label_id)

    assert almost_equal(scores["custom_weighted_mean_recall"]["mean"], 0.5)
    assert almost_equal(scores["custom_weighted_mean_precision"]["mean"], 0.5)
    assert almost_equal(scores["custom_weighted_mean_f1"]["mean"], 0.5)


def test_remove_label_inplace_True(review_set):
    assert sorted(review_set.get_all_label_ids()) == [
        "bp-golden_v2",
        "openai-chat_gpt_4_matthisv1-test",
    ]

    review_set.remove_label("openai-chat_gpt_4_matthisv1-test", inplace=True)

    assert sorted(review_set.get_all_label_ids()) == [
        "bp-golden_v2",
    ]


def test_remove_label_inplace_False(review_set):
    label_ids_before = ["bp-golden_v2", "openai-chat_gpt_4_matthisv1-test"]
    label_ids_after = ["bp-golden_v2"]

    assert sorted(review_set.get_all_label_ids()) == label_ids_before

    review_set_without_label = review_set.remove_label(
        "openai-chat_gpt_4_matthisv1-test", inplace=False
    )

    assert sorted(review_set.get_all_label_ids()) == label_ids_before
    assert sorted(review_set_without_label.get_all_label_ids()) == label_ids_after
