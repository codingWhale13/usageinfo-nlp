from helpers.review_set import ReviewSet

# This script was needed because I annotated with models using "," as separator but the generation also got upgraded to ";"-separation

PATH_IN = "nils_ba/experiment_float16/annotations/val-float-all.json"
PATH_OUT = "nils_ba/experiment_float16/annotations/val-float-all-vX.json"

rs = ReviewSet.from_files(PATH_IN)
for review in rs:
    for label_id in rs.get_all_label_ids():
        usage_options = review.get_usage_options(label_id)
        if len(usage_options) == 0:
            continue

        if len(usage_options) > 1 and "gpt" not in label_id:
            print("HOW DID THIS HAPPEN?", label_id, usage_options)

        new_usage_options = []
        for uo in usage_options:
            new_usage_options.extend(uo.split(","))
        review.data["labels"][label_id]["usageOptions"] = new_usage_options

rs.save(PATH_OUT)
