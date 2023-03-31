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


def test_get_item(reviews):
    pass


def test_equality(reviews):
    for review in reviews:
        assert review == review


def test_key(reviews):
    pass
