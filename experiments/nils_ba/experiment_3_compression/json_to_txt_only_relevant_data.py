from helpers.review_set import ReviewSet

FILE_IN = "train.json"
FILE_OUT = "train_usage_options.txt"

rs = ReviewSet.from_files(FILE_IN)

with open(FILE_OUT, "w") as file:
    for review in rs:
        file.write(f"{';'.join(review.get_usage_options('chat_gpt_clean'))}\n")
