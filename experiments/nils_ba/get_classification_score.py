from helpers.review_set import ReviewSet

PATH = "data/ba/ba-30k-val-all-v2.json"

rs = ReviewSet.from_files(PATH)

print(rs.get_all_label_ids())
for label_id in rs.get_all_label_ids():
    if not label_id.endswith("config_test"):
        continue
    tp = tn = fp = fn = 0
    for review in rs:
        usage_predicted = len(review.get_usage_options(label_id)) > 0
        usage_ground_truth = len(review.get_usage_options("chat_gpt_clean")) > 0
        if usage_predicted == usage_ground_truth == True:
            tp += 1
        elif usage_predicted == usage_ground_truth == False:
            tn += 1
        elif usage_predicted != usage_ground_truth == True:
            fn += 1
        else:
            fp += 1
    print(label_id, tp, tn, fp, fn, (tp + tn) / (tp + tn + fp + fn))
