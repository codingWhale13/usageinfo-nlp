import argparse

from clusterer import Clusterer
from data_loader import DataLoader
from scorer import Scorer

import helpers.label_selection as ls
import utils
from helpers.review_set import ReviewSet


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Perform clustering on the given reviewsets"
    )
    parser.add_argument(
        "reviewset_files",
        type=str,
        nargs="+",
        help="All reviewset files to cluster usage options from",
    )
    parser.add_argument(
        "label_ids",
        type=str,
        help="Which label(s) (aka which usage options) to use for clustering",
    )
    parser.add_argument(
        "-c",
        "--clustering_config",
        type=str,
        default="clustering_config",
        help="Clustering config to use",
    )

    return parser.parse_args(), parser.format_help()


def main():
    args, _ = arg_parse()
    label_ids = args.label_ids.split(", ")
    files = args.reviewset_files
    review_set = ReviewSet.from_files(*files)
    clustering_config = utils.get_config(args.clustering_config)
    review_set_df, df_to_cluster = DataLoader(
        review_set, ls.LabelIDSelectionStrategy(*label_ids), clustering_config["data"]
    ).load()

    scores = {}
    arg_dicts = utils.get_arg_dicts(clustering_config, len(review_set_df))

    for arg_dict in arg_dicts:
        print(f"Clustering with {arg_dict}...")
        clustered_df = Clusterer(df_to_cluster, arg_dict).cluster()

        if clustering_config["data"]["n_components"] == 2:
            utils.plot_clusters2d(
                clustered_df, arg_dict, color="label", interactive=True
            )

        print(f"Scoring clustering with {arg_dict}...")
        scores[[v for v in arg_dict.values() if v is not None][0]] = Scorer(
            clustered_df
        ).score()

        if clustering_config["evaluation"]["merge_duplicates"]:
            clustered_df = utils.merge_duplicated_usage_options(
                clustered_df, review_set_df
            )

        if clustering_config["evaluation"]["save_to_disk"]:
            utils.save_clustered_df(clustered_df, arg_dict)

    utils.plot_scores(scores, "scores.png")


if __name__ == "__main__":
    main()
