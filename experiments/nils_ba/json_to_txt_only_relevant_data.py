from helpers.review_set import ReviewSet

FILE_IN = "data/ba/ba-30k-val-all.json"
FILE_OUT = "data/ba/ba-30k-val-all.txt"

rs = ReviewSet.from_files(FILE_IN)

with open(FILE_OUT, "w") as file:
    for review in rs:
        file.write(f"Review headline: {review.data['review_headline']}\n")
        file.write(f"Review body: {review.data['review_body']}\n")
        # TODO: add labels?


