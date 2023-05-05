from data_loader import DataLoader
from clusterer import Clusterer
from scorer import Scorer
import argparse
import utils


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Perform clustering on the given reviewsets."
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
        help="Which label (aka which usage otpions) to use for clustering",
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

    usage_options, embedded_usage_options = DataLoader(
        file_paths, label_id, clustering_config["data"]
    ).load()

    if len(usage_options) < max(clustering_config["clustering"]["n_clusters"]):
        raise ValueError(
            f"Number of usage options ({len(usage_options)}) is smaller than the maximum number of clusters ({max(clustering_config['clustering']['n_clusters'])})"
        )
    scores = {}

    for n_clusters in clustering_config["clustering"]["n_clusters"]:
        print(f"Clustering with {n_clusters} clusters...")
        labels, centroids = Clusterer(
            embedded_usage_options, clustering_config["clustering"], n_clusters
        ).cluster()

        scores[n_clusters] = Scorer(embedded_usage_options, labels, centroids).score()

    utils.plot_scores(scores, "scores.png")

    # TODO: Save scores/clusters to file or display in some way


if __name__ == "__main__":
    main()
