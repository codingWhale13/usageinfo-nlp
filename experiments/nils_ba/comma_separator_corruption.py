from helpers.review_set import ReviewSet

for subset in ["train", "val", "test"]:
    rs = ReviewSet.from_files(
        f"/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/30k_baseline/{subset}.json"
    )
    comma_count = 0
    for review in rs:
        contains_comma = False
        for usage_option in review.get_label_for_id(
            "bp-chat_gpt_correction", "chat_gpt-vanilla-baseline"
        )["usageOptions"]:
            if "," in usage_option:
                contains_comma = True
                break
        if contains_comma:
            comma_count += 1
    print(
        f"Corrupted reviews in ba-30k-{subset} because comma exists in usage option: {comma_count}/{len(rs)}"
    )

"""
RESULT:
Corrupted reviews in ba-30k-train because comma exists in usage option: 95/20784
Corrupted reviews in ba-30k-val because comma exists in usage option: 15/3167
Corrupted reviews in ba-30k-test because comma exists in usage option: 26/5979
"""
