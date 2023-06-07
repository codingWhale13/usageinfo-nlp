import argparse

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
        "label_id",
        type=str,
        help="Which label (aka which usage options) to use for clustering",
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
    label_id = args.label_id
    file_paths = args.reviewset_files
    clustering_config = utils.get_config(args.clustering_config)
    review_set_df = DataLoader(file_paths, label_id, clustering_config["data"]).load()

    scores = {}
    arg_dicts = utils.get_arg_dicts(clustering_config, len(review_set_df))

    for arg_dict in arg_dicts:
        print(f"Clustering with {arg_dict}...")
        clustered_df = Clusterer(
            review_set_df,
            clustering_config["clustering"],
            **arg_dict,
        ).cluster()

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

    if clustering_config["data"]["n_components"] == 2:
        utils.plot_clusters2d(
            clustered_df, arg_dict, color="product_category", interactive=True
        )
    utils.plot_scores(scores, "scores.png")


if __name__ == "__main__":
    main()
