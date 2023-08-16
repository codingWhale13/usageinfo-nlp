# %%
from helpers.review_set import ReviewSet
from helpers.review import Review
import math

review_attributes = {"product_category", "score", "review_body"}
review_data = {
    "2": {"score": 2.0, "product_category": "Tv", "review_body": "Bad tv"},
    "home-1": {
        "score": 0.5,
        "product_category": "Home",
        "review_body": "This was a great home",
    },
    "home-2": {
        "score": 0.5,
        "product_category": "Home",
        "review_body": "This is a great home",
    },
    "hom2-3": {
        "score": 0.5,
        "product_category": "Home",
        "review_body": "This was a great home",
    },
    "home-4": {
        "score": 0.5,
        "product_category": "Home",
        "review_body": "This was a great home",
    },
    "home-5": {
        "score": 0.5,
        "product_category": "Home",
        "review_body": "This was a great home",
    },
    "4": {
        "score": 4.2,
        "review_body": "Did not work. Did not help me",
        "product_category": "Food",
    },
    "garden-1": {
        "score": 1.1,
        "review_body": "Helped with yard work",
        "product_category": "Garden",
    },
    "garden-2": {
        "score": 1.0,
        "review_body": "Great yard work",
        "product_category": "Garden",
    },
}

reviews = []
uncertainity_scores = {}
for review_id, data in review_data.items():
    review = Review(review_id, data)
    review.review_attributes = review_attributes
    reviews.append(review)
    uncertainity_scores[review_id] = data["score"]

review_set = ReviewSet.from_reviews(*reviews)
# print(review_set.reviews.keys(), uncertainity_scores)

%load_ext autoreload
%autoreload 2
from active_learning.sampler import GreedyOptimalClusterSubsetSampler

sampler = GreedyOptimalClusterSubsetSampler(n=5)
sample, score = sampler.sample(review_set, uncertainity_scores)
sample.reviews.keys(), score
# %%
assert set(sample.reviews.keys()) == {
    "2",
    "home-1",
    "4",
    "garden-1",
    "garden-2",
}, "Sample 1 not correct"

sampler = GreedyOptimalClusterSubsetSampler(n=2)
sample, score = sampler.sample(review_set, uncertainity_scores)

assert set(sample.reviews.keys()) == {
    "4",
    "garden-1",
}, f"Sample 2 not correct: {sample.reviews.keys()}"
