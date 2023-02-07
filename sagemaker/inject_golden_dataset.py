import random

def mix_normal_and_golden_samples(
    sample_json, golden_json, golden_fraction, total_output_reviews
):
    random.shuffle(sample_json)
    random.shuffle(golden_json)

    reduced_json = sample_json[:total_output_reviews]

    # every `step_size`th review will be replaced by a golden review s.t. the golden fraction is met
    step_size = round(1 / golden_fraction)

    print(
        f"Replacing every {step_size}th review (~{golden_fraction * 100}%) with a golden review..."
    )
    if len(reduced_json) / step_size > len(golden_json):
        print(f"WARNING: Some golden labels will be used more than once.")

    golden_idx = 0
    for idx_to_replace in range(0, len(reduced_json), step_size):
        golden_review = golden_json[golden_idx]
        reduced_json[idx_to_replace] = golden_review
        print(f"Inserted golden review: {golden_review['product_title']}")
        golden_idx = (golden_idx + 1) % len(golden_json)

    return reduced_json