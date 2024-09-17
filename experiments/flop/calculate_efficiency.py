#!/usr/bin/env python3

import pandas as pd


TRAINING_FLOP = 13675914923606016
LABELED_DATAPOINTS = 2000
FLOP_PER_REQUEST_T5 = round(205962234101.76)
TOKENS_PER_REQUEST_T5 = 23840 / 4000

# Starting at 10 TFLOP per forward pass
gpt4_flop_per_request_estimates = [
    (int)(TOKENS_PER_REQUEST_T5 * (10**x)) for x in range(13, 17)
]

num_requests = [(int)(10 ** (x / 2)) for x in range(4, 25)]


def total_flop_t5(requests: int, flop_per_request_gpt4) -> int:
    fine_tuning_flop = TRAINING_FLOP + LABELED_DATAPOINTS * flop_per_request_gpt4
    inference_flop = requests * FLOP_PER_REQUEST_T5
    return fine_tuning_flop + inference_flop


def total_flop_gpt4(requests: int, flop_per_request_gpt4) -> int:
    return requests * flop_per_request_gpt4


labels = ["10 TFLOP", "100 TFLOP", "1 PFLOP", "10 PFLOP"]
df_t5 = pd.DataFrame(num_requests, columns=["Requests"])
df_gpt4 = pd.DataFrame(num_requests, columns=["Requests"])

for label, flop_per_request_gpt4 in zip(labels, gpt4_flop_per_request_estimates):
    df_t5[label] = df_t5["Requests"].apply(
        lambda x: total_flop_t5(x, flop_per_request_gpt4)
    )
    df_gpt4[label] = df_gpt4["Requests"].apply(
        lambda x: total_flop_gpt4(x, flop_per_request_gpt4)
    )

df_diff = df_t5 - df_gpt4
df_diff["Requests"] = df_t5["Requests"]

print(df_t5)
print(df_gpt4)
print(df_diff)
# print(FLOP_PER_REQUEST_T5 / TOKENS_PER_REQUEST_T5)
