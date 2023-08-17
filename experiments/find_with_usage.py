from helpers.review_set import ReviewSet

FILE = "../../playground/sample_1398494-12.json"
LABEL_ID = "model-flan-t5-base_42-find_sample"


def main():
    review_set = ReviewSet.from_files(FILE)
    reviews_with_usage_options = []
    for review in review_set:
        prediction_has_usage_options = (
            len(review.get_label_for_id(LABEL_ID)["usageOptions"]) > 0
        )
        if prediction_has_usage_options:
            reviews_with_usage_options.append(review.review_id)

    print(len(reviews_with_usage_options))

    review_set.filter(lambda review: review.review_id in reviews_with_usage_options)
    review_set.save("sample_1398494-12_with_usage_options.json")


if __name__ == "__main__":
    main()
