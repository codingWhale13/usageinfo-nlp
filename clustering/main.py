import argparse

from helpers.review_set import ReviewSet
from clusterer import Clusterer
from data_loader import DataLoader
from scorer import Scorer

import utils


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
        nargs="+",
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
    label_ids = args.label_ids
    file_paths = args.reviewset_files
    review_set = ReviewSet.from_files(file_paths)
    clustering_config = utils.get_config(args.clustering_config)
    review_set_df = DataLoader(review_set, label_ids, clustering_config["data"]).load()

    scores = {}
    arg_dicts = utils.get_arg_dicts(clustering_config, len(review_set_df))

    for arg_dict in arg_dicts:
        print(f"Clustering with {arg_dict}...")
        clustered_df = Clusterer(review_set_df, arg_dict).cluster()

        if clustering_config["data"]["n_components"] == 2:
            utils.plot_clusters2d(
                clustered_df, arg_dict, color="label", interactive=True
            )

        if clustering_config["clustering"]["save_to_disk"]:
            utils.save_clustered_df(clustered_df, arg_dict)

        print(f"Scoring clustering with {arg_dict}...")
        scores[[f"{v}" for v in arg_dict.values() if v is not None][0]] = Scorer(
            clustered_df
        ).score()

    utils.plot_scores(scores, "scores.png")


if __name__ == "__main__":
    main()
