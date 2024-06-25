import tiktoken
import sys

from helpers.review_set import ReviewSet
from helpers.label_selection import LabelIDSelectionStrategy

enc = tiktoken.encoding_for_model("gpt-4")


LABEL_ID = "gpt_4-vanilla_6-paper_trainset"
reviews = ReviewSet.from_files(sys.argv[1])

tokens = 0
num_reviews = 0
token_counts = []
for review in reviews:
    label = review.get_label_from_strategy(LabelIDSelectionStrategy(LABEL_ID))
    if not label:
        continue
    num_reviews += 1
    t = len(enc.encode(label["metadata"]["text_completion"]))
    tokens += t
    token_counts.append(t)

print(f"Average tokens per request: {tokens / num_reviews}")
print(f"Median tokens per request: {sorted(token_counts)[len(token_counts) // 2]}")
print(f"(Number of reviews: {num_reviews})")
print(f"(Number of tokens: {tokens})")
