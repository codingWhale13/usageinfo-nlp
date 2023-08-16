# %%
from training.probablistic_generator import BatchProbabilisticGenerator
from helpers.review_set import ReviewSet
from copy import deepcopy
import pandas as pd
from helpers.sustainability_tracker import SustainabilityTracker

BASE_HYPERPARAMERTERS = {
    "batch_size": 64,
    "max_sequence_length": 64,
    "max_iterations": 1280,
    "minimum_probability": 0.01,
    "minimum_total_probability": 0.95,
    "token_top_k": 5,
}


review_set_name = "ba-30k-train-reviews.json"  # "ba-30k-train-reviews.json"
reviews = ReviewSet.from_files(review_set_name)

parameter_id = 1
stats_df = []


def benchmark(generator_parameters: dict):
    global parameter_id
    prob_generator = BatchProbabilisticGenerator(
        prompt_id="active_learning_v1",
        artifact_name="puppy-barfs-adorable-98-3",
        checkpoint="best",
        **generator_parameters,
    )

    results = prob_generator.generate_usage_options_prob_based_batch(
        reviews, cluster_results=False
    )

    review_stats = {}
    stats = {
        "total_reviews": 0,
        "reviews_that_reached_max_iterations": 0,
        "generation_reached_max_length": 0,
        "generation_has_too_low_branch_probability": 0,
        "generation_finished": 0,
        "total_probability": 0,
        "total_generations": 0,
        "unfinished_probability": 0,
    }

    for review_id, review in results.items():
        review_stats[review_id] = {"generation_finished": 0, "review_id": review_id}
        review_stats[review_id]["generation_reached_max_length"] = 0
        review_stats[review_id]["total_probability"] = 0
        review_stats[review_id]["generation_has_too_low_branch_probability"] = 0
        review_stats[review_id]["unfinished_probability"] = 0
        review_stats[review_id]["generation_has_too_low_branch_probability"] = 0
        for x in review:
            stats["total_generations"] += 1
            if (
                len(x["decoder_token_ids"])
                == generator_parameters["max_sequence_length"]
            ):
                review_stats[review_id]["generation_reached_max_length"] += 1
                stats["generation_reached_max_length"] += 1
                stats["total_probability"] += x["probability"]
                review_stats[review_id]["total_probability"] += x["probability"]
            elif 1 in x["decoder_token_ids"]:
                review_stats[review_id]["generation_finished"] += 1
                stats["generation_finished"] += 1
                stats["total_probability"] += x["probability"]
                review_stats[review_id]["total_probability"] += x["probability"]
            else:
                pass
                #review_stats[review_id]["total_probability"] += x["probability"]
                #stats["generation_has_too_low_branch_probability"] += 1
                #stats["unfinished_probability"] += x["probability"]
                #review_stats[review_id]["unfinished_probability"] += x["probability"]
                #review_stats[review_id][
                #    "generation_has_too_low_branch_probability"
                #] += 1

        stats["total_reviews"] += 1
        # if iterations >= generator_parameters["max_iterations"]:
        #    stats["reviews_that_reached_max_iterations"] += 1

    stats["mean_total_probability"] = (
        stats["total_probability"] / stats["total_reviews"]
    )
    stats_df.append(stats | generator_parameters | {"parameter_id": parameter_id})
    pd.DataFrame.from_records(stats_df).to_csv(f"pg_hp_test_{parameter_id}_2.csv")
    pd.DataFrame.from_records(list(review_stats.values())).to_csv(
        f"reviews_pg_hp_test_{parameter_id}_2.csv"
    )
    print(generator_parameters, parameter_id)
    print(stats)
    print(review_stats)
    parameter_id += 1


def sweep(key, values, sustainability_tracker: SustainabilityTracker):
    for value in values:
        sustainability_tracker.start(f"{key}:{value}")
        benchmark(deepcopy(BASE_HYPERPARAMERTERS) | {key: value})
        sustainability_tracker.stop(f"{key}:{value}")

        sustainability_tracker.results_to_dataframe().to_csv(
            "emissions_pg_benchmark_2.csv"
        )


with SustainabilityTracker() as sustainability_tracker:
    sweep("token_top_k", [1, 2, 5, 10, 20, 50], sustainability_tracker)

    sweep(
        "minimum_probability",
        [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        sustainability_tracker,
    )
    sweep("max_sequence_length", [8, 16, 32, 64, 128], sustainability_tracker)
