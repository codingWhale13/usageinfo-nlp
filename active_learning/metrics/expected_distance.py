# %%

results = [
    {"probability": 0.25, "decoder_input_ids": [123, 115, 237, 125]},
    {"probability": 0.45, "decoder_input_ids": [123, 115, 136, 125]},
    {"probability": 0.3, "decoder_input_ids": [224, 216, 237, 226]},
]
results

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NO_USAGE_OPTIONS_DECODER_IDS = [0, 465, 169, 1488, 1]


def similarity_matrix(decoder_input_ids: list[list[int]]):
    def tokenize(sentences):
        return sentences

    count_vect = CountVectorizer(
        tokenizer=tokenize, lowercase=False, token_pattern=None
    )
    X_train_counts = count_vect.fit_transform(decoder_input_ids)
    cosine_similarity_matrix = cosine_similarity(X_train_counts.toarray())
    print("before", cosine_similarity_matrix)
    for i in range(len(decoder_input_ids)):
        if decoder_input_ids[i] == NO_USAGE_OPTIONS_DECODER_IDS:
            cosine_similarity_matrix[i, :] = np.array([0] * len(decoder_input_ids))
            cosine_similarity_matrix[:, i] = np.array([0] * len(decoder_input_ids))
            cosine_similarity_matrix[i][i] = 1

    return cosine_similarity_matrix


def score(results):
    distance_function = 1 - similarity_matrix([x["decoder_input_ids"] for x in results])

    def score_one(i, results, distance_function):
        return sum(
            [
                results[j]["probability"] * distance_function[i][j]
                for j in range(len(results))
            ]
        )

    return sum(
        [
            results[i]["probability"] * score_one(i, results, distance_function)
            for i in range(len(results))
        ]
    )


import numpy as np


def score_vector(results):
    distance_function = similarity_matrix([x["decoder_input_ids"] for x in results])
    print(distance_function)
    probs = np.array([x["probability"] for x in results])
    return 1 - np.matmul(np.matmul(probs.T, distance_function), probs.T)


score_vector(results)  # , score(results)
# %%
import time

normal = []
vec = []

for _ in range(1000):
    start = time.perf_counter()
    score(results)
    normal.append(time.perf_counter() - start)

for _ in range(1000):
    start = time.perf_counter()
    score_vector(results)
    vec.append(time.perf_counter() - start)

from statistics import mean

print(mean(normal), mean(vec))
