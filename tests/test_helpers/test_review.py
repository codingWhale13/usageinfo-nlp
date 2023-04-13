from copy import copy, deepcopy
import json
import pytest

from helpers.review_set import ReviewSet
from helpers.review import Review


@pytest.fixture
def reviews():
    with open("tests/test_helpers/mock_review.json") as file:
        json_data = json.load(file)
        reviews = []
        for review_id, review_data in json_data["reviews"].items():
            reviews.append(Review(review_id, review_data))
        return reviews


def test_equality(reviews):
    assert reviews[0] != reviews[1]

    for review in reviews:
        assert review == review


def test_or(reviews):
    review_1, review_2 = reviews
    with pytest.raises(AssertionError):
        review_1 | review_2  # reviews don't have the same ID

    label_a, label_b = "NEW_LABEL_A", "NEW_LABEL_B"
    review_1_with_label_a = deepcopy(review_1)
    for label_id in review_1_with_label_a.get_label_ids():
        review_1_with_label_a.remove_label(label_id)
    review_1_with_label_b = deepcopy(review_1_with_label_a)
    review_1_with_label_a.add_label(label_a, ["some usage option A"])
    review_1_with_label_b.add_label(label_b, ["some usage option B"])

    merged_review = review_1_with_label_a | review_1_with_label_b
    assert sorted(merged_review.get_label_ids()) == [label_a, label_b]


def test_ior(reviews):
    review_1, review_2 = reviews
    with pytest.raises(AssertionError):
        review_1 |= review_2  # reviews don't have the same ID

    label_a, label_b = "NEW_LABEL_A", "NEW_LABEL_B"
    review_1_with_label_a = deepcopy(review_1)
    for label_id in review_1_with_label_a.get_label_ids():
        review_1_with_label_a.remove_label(label_id)
    review_1_with_label_b = deepcopy(review_1_with_label_a)
    review_1_with_label_a.add_label(label_a, ["some usage option A"])
    review_1_with_label_b.add_label(label_b, ["some usage option B"])

    merged_review = review_1_with_label_a
    merged_review |= review_1_with_label_b
    assert sorted(merged_review.get_label_ids()) == [label_a, label_b]


def test_copy(reviews):
    review = reviews[0]
    copied_review = copy(review)

    assert review == copied_review  # review ID should be the same

    assert id(review) != id(copied_review)  # they should be two different objects
    # strings don't want to be copied... https://www.youtube.com/watch?v=g6aJhYM0Kc0
    assert id(review.review_id) == id(copied_review.review_id)
    assert id(review.data) != id(copied_review.data)  # data was shallow copied
    assert id(review.data["labels"]) == id(copied_review.data["labels"])  # shallow!

    copied_review.review_id = "some other review id"
    assert review != copied_review

    copied_review.add_label("my_label", ["some usage option"])
    assert len(review.get_label_ids()) == 3  # side effects! (because of shallow copy)
    assert len(copied_review.get_label_ids()) == 3


def test_deepcopy(reviews):
    review = reviews[0]
    copied_review = deepcopy(review)

    assert review == copied_review  # review ID should be the same

    assert id(review) != id(copied_review)  # they should be two different objects
    # strings don't want to be copied... https://www.youtube.com/watch?v=g6aJhYM0Kc0
    assert id(review.review_id) == id(copied_review.review_id)
    assert id(review.data) != id(copied_review.data)  # data was deep copied
    assert id(review.data["labels"]) != id(copied_review.data["labels"])  # deep!

    copied_review.review_id = "some other review id"
    assert review != copied_review

    copied_review.add_label("my_label", ["some usage option"])
    assert len(review.get_label_ids()) == 2
    assert len(copied_review.get_label_ids()) == 3


def test_get_labels(reviews):
    review = reviews[0]
    assert review.get_labels() == review.data["labels"]


def test_get_label_ids(reviews):
    review = reviews[0]
    assert sorted(review.get_labels()) == sorted(review.data["labels"].keys())


def test_get_label(reviews):
    review = reviews[0]
    assert (
        review.get_label_for_id("bp-golden_v2") == review.data["labels"]["bp-golden_v2"]
    )


def test_remove_label_inplace_True(reviews):
    for review in reviews:
        assert sorted(review["labels"].keys()) == [
            "bp-golden_v2",
            "openai-chat_gpt_4_matthisv1-test",
        ]

        review.remove_label("bp-golden_v2", inplace=True)

        assert sorted(review["labels"].keys()) == ["openai-chat_gpt_4_matthisv1-test"]


def test_remove_label_inplace_False(reviews):
    label_ids_before = [
        "bp-golden_v2",
        "openai-chat_gpt_4_matthisv1-test",
    ]
    label_ids_after = ["openai-chat_gpt_4_matthisv1-test"]

    for original_review in reviews:
        assert sorted(original_review["labels"].keys()) == label_ids_before

        review_without_label = original_review.remove_label(
            "bp-golden_v2", inplace=False
        )

        assert sorted(original_review["labels"].keys()) == label_ids_before
        assert sorted(review_without_label["labels"].keys()) == label_ids_after
