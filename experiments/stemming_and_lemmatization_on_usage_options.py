from helpers.review_set import ReviewSet

path = "./data/comparison_easy_names.json"

rs = ReviewSet.from_files(path)

golden_label = "Golden"
for label_id in rs.get_all_label_ids():
    if label_id != golden_label:
        agg = rs.get_agg_scores(
            label_id,
            golden_label,
            [
                "custom_weighted_mean_f1",
                "custom_weighted_mean_f1_stem",
                "custom_weighted_mean_f1_lemmatize",
            ],
        )
        print(f"{label_id} vs. {golden_label}:")
        for metric_id, metric_value in agg.items():
            print(f"{metric_id}: {metric_value['mean']}")
        print()

rs.save_as(path)

"""
RESULT:

ChatGPT vs. Golden:
custom_weighted_mean_f1: 0.7342848392428813
custom_weighted_mean_f1_stem: 0.7221824236210207
custom_weighted_mean_f1_lemmatize: 0.7340971194553267

Our Model vs. Golden:
custom_weighted_mean_f1: 0.6887454054085349
custom_weighted_mean_f1_stem: 0.6764002607050734
custom_weighted_mean_f1_lemmatize: 0.6882922980935972

GPT4 vs. Golden:
custom_weighted_mean_f1: 0.7332017248610293
custom_weighted_mean_f1_stem: 0.7289783635001361
custom_weighted_mean_f1_lemmatize: 0.736989312089465

Take-Away: Stemming and Lemmatization did not have the desired effect of improving scores :/
"""
