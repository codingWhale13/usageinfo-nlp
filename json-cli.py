#!/usr/bin/env python3
import argparse
import os
import copy
import pprint
import fnmatch

from helpers.review_set import ReviewSet

dash = "-" * 80


class bcolors:
    BLUE = "\033[94m"
    RED = "\033[91m"
    DARKYELLOW = "\033[33m"
    GREEN = "\033[92m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Easily handle reviewset json files. Call the script with only a json file path to see some stats about that reviewset."
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    stats_parser = subparsers.add_parser(
        "stats",
        help="Just print some stats about the reviewset",
    )
    stats_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    stats_parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save the base reviewset (only useful for auto-upgrading)",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge reviews and/or labels of other files into the base reviewset",
    )
    merge_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    merge_parser.add_argument(
        "merge_files",
        type=str,
        nargs="+",
        metavar="merge_file",
        help="Reviewset file(s) to merge into the base file",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract one or more label(s) from the base reviewset to a new file",
    )
    extract_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    extract_parser.add_argument(
        "out_file",
        type=str,
        help="Filepath to save the extracted reviews to",
    )
    extract_parser.add_argument(
        "label_ids",
        type=str,
        nargs="+",
        metavar="label_id",
        help="Label(s) to extract from the base file (wildcard labels need to be in quotes)",
    )
    extract_parser.add_argument(
        "--keep",
        "-k",
        action="store_true",
        help="Keep all reviews, even those without the specified label(s)",
    )

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete one or more label(s) from the base reviewset",
    )
    delete_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    delete_parser.add_argument(
        "label_ids",
        type=str,
        nargs="+",
        metavar="label_id",
        help="Label(s) to delete from the base file (wildcard labels need to be in quotes)",
    )
    delete_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Don't ask for confirmation before deleting the label(s)",
    )

    sample_parser = subparsers.add_parser(
        "sample",
        help="Sample some reviews from the base reviewset and save them to a new file or print them (or both)",
    )
    sample_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    sample_parser.add_argument(
        "n",
        type=int,
        nargs="?",
        help="Number of reviews to sample from the base reviewset",
    )
    sample_parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="out_file",
        help="Filepath to save the cut reviews to",
    )
    sample_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress the printing of the sampled reviews",
    )
    sample_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Sample all reviews from the base reviewset",
    )
    sample_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Seed to use for sampling (default is None)",
    )

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotate the base reviewset with a trained model artifact",
    )
    annotate_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    annotate_parser.add_argument(
        "artifact_name",
        type=str,
        metavar="model_artifact_name",
        help="Name of the model artifact to use for annotation",
    )
    annotate_parser.add_argument(
        "last_part_of_label_id",
        type=str,
        nargs="?",
        default=None,
        metavar="last_part_of_label_id",
        help="Last part (aka the unique identifier) of the label_id that the annotation should be (optionally) saved under",
    )
    annotate_parser.add_argument(
        "--checkpoint",
        "-c",
        type=int,
        help="Optional checkpoint of the artifact to use for annotation (default is the last checkpoint)",
    )
    annotate_parser.add_argument(
        "--generation_config",
        "-g",
        type=str,
        default=None,
        help="Generation config to use for the annotation (default is defined in the generator)",
    )
    annotate_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output of the annotation"
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score labels against each other",
    )
    score_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    score_parser.add_argument(
        "reference_label_candidate_ids",
        type=str,
        # bug in argparse lol, therefore (()) instead of ()
        metavar="reference_label_id((s))",
        help="Reference label(s) to score against, multiple candidates must be separated by a comma (also, wildcard labels need to be in quotes)",
    )
    score_parser.add_argument(
        "label_ids",
        type=str,
        metavar="label_id",
        nargs="*",
        help="Label(s) to score (wildcard labels need to be in quotes)",
    )
    score_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Score all other labels against the reference label(s)",
    )
    score_parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        help="Metrics to use for scoring (comma separated, default is defined in scoring/__init__.py)",
    )

    remove_outliers_parser = subparsers.add_parser(
        "remove_outliers",
        help="Use clustering to remove outlier usage options from a label",
    )
    remove_outliers_parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    remove_outliers_parser.add_argument(
        "label_id",
        type=str,
        metavar="label_id",
        help="Label to remove outliers from",
    )
    remove_outliers_parser.add_argument(
        "output_file",
        type=str,
        help="Filepath to save the reviewset with removed outliers to",
    )
    remove_outliers_parser.add_argument(
        "--dist_threshold",
        "-d",
        type=float,
        default=0.2,
        help="Distance threshold for clustering (default: 0.2)",
    )
    remove_outliers_parser.add_argument(
        "--keep",
        "-k",
        type=float,
        default=0.8,
        help="Percentage (float between 0 and 1) of usage options to keep (rest will be removed) [default: 0.8]",
    )
    return parser.parse_args(), parser.format_help()


def print_stats(reviewset: ReviewSet, prefix: str = ""):
    label_ids = list(reviewset.get_all_label_ids())
    label_ids_with_count = [
        (label_id, len(reviewset.reviews_with_labels({label_id})))
        for label_id in label_ids
    ]

    print(
        f"{prefix}The reviewset contains {bcolors.BLUE}{len(reviewset)}{bcolors.ENDC} reviews and the following {bcolors.BLUE}{len(reviewset.get_all_label_ids())}{bcolors.ENDC} label(s):\n{prefix}\t",
        dash + f"\n{prefix}\t",
        "{1:<40}{2:<40}\n{0}\t".format(prefix, "Label", "Coverage"),
        dash + f"\n{prefix}\t",
        f"\n{prefix}\t".join(
            f"{label_id:<40}{str(count) + '/' + str(len(reviewset)):<40}"
            for label_id, count in label_ids_with_count
        ),
        sep="",
    )


def filter_invalid_labels(reviewset: ReviewSet, label_ids: list) -> list:
    all_label_ids = list(reviewset.get_all_label_ids())

    result = []
    for label_id in label_ids:
        if "*" in label_id:
            matches = fnmatch.filter(all_label_ids, label_id)
            if matches:
                for match in matches:
                    result.append(match)
            else:
                print(
                    f"\nNo matching label found for wildcard {bcolors.DARKYELLOW}{label_id}{bcolors.ENDC}, skipping..."
                )
        else:
            if label_id in all_label_ids:
                result.append(label_id)
            else:
                print(
                    f"\nLabel {bcolors.DARKYELLOW}{label_id}{bcolors.ENDC} does not exist in the base reviewset, skipping..."
                )

    return result


def stats(base_reviewset: ReviewSet, args: argparse.Namespace):
    if args.save:
        base_reviewset.save()
        print(f"\n{bcolors.GREEN}Reviewset saved!{bcolors.ENDC}")


def merge(base_reviewset: ReviewSet, args: argparse.Namespace):
    from helpers.label_selection import LabelIDSelectionStrategy

    print(f"\n\nMerging {len(args.merge_files)} file(s) into base file")
    for counter, reviewset_file in enumerate(args.merge_files):
        reviewset = ReviewSet.from_files(reviewset_file)
        print(
            f"\n[{counter + 1}] Merging file: {bcolors.BLUE}{reviewset.save_path}{bcolors.ENDC}"
        )
        print_stats(reviewset, prefix="\t")

        new_review_count = base_reviewset.count_new_reviews(reviewset)
        allow_new_reviews = True
        if new_review_count == 0:
            print(f"\n\t{bcolors.DARKYELLOW}No new reviews found{bcolors.ENDC}")
        else:
            print(
                f"\n\tThe reviewset contains {bcolors.BLUE}{new_review_count}{bcolors.ENDC} new reviews."
            )
            if (
                input(
                    "\tDo you want to add them to the base reviewset when merging labels? [Y/n]: "
                ).lower()
                == "n"
            ):
                allow_new_reviews = False

        new_label_ids = (
            reviewset.get_all_label_ids() - base_reviewset.get_all_label_ids()
        )
        merge_label_ids_with_base_count = {}
        for label_id in reviewset.get_all_label_ids():
            filtered_reviewset = reviewset.filter_with_label_strategy(
                LabelIDSelectionStrategy(label_id), inplace=False
            )
            merge_label_ids_with_base_count[
                label_id
            ] = base_reviewset.count_common_reviews(filtered_reviewset)

        if not new_label_ids:
            print(f"\n\t{bcolors.DARKYELLOW}No new labels found.{bcolors.ENDC}")
            if not allow_new_reviews or new_review_count == 0:
                print(
                    f"\n\tIt might still make sense to merge if, for example, the new reviewset has a better coverage of some existing labels in the base reviewset."
                )
        else:
            print(
                f"\n\tFor the following {bcolors.BLUE}{len(new_label_ids)}{bcolors.ENDC} new label(s), that many reviews labelled by them could also be found in the base reviewset:\n",
                "\t\t" + dash + "\n\t\t",
                "{0:<40}{1:<40}\n\t\t".format(
                    "Label", "Coverage of reviews in base file"
                ),
                dash + f"\n\t\t",
                f"\n\t\t".join(
                    f"{label_id:<40}{str(merge_label_ids_with_base_count[label_id]) + '/' + str(len(base_reviewset)):<40}"
                    for label_id in new_label_ids
                ),
                sep="",
            )

        if input("\n\tDo you want to merge the file? [Y/n]: ").lower() == "n":
            print(f"\n\t{bcolors.DARKYELLOW}Aborted!{bcolors.ENDC}")
            continue

        base_reviewset.merge(
            reviewset, allow_new_reviews=allow_new_reviews, inplace=True
        )
        base_reviewset.save()
        print(f"\n\t{bcolors.GREEN}Merged!{bcolors.ENDC}")


def extract(base_reviewset: ReviewSet, args: argparse.Namespace):
    from helpers.label_selection import LabelIDSelectionStrategy

    extract_label_ids = filter_invalid_labels(base_reviewset, args.label_ids)
    if not extract_label_ids:
        print(f"\n{bcolors.DARKYELLOW}No labels to extract, aborting...{bcolors.ENDC}")
        return

    remove_label_ids = list(base_reviewset.get_all_label_ids() - set(extract_label_ids))

    if not args.keep:
        label_selection_strategy = LabelIDSelectionStrategy(*extract_label_ids)
        base_reviewset.filter_with_label_strategy(
            label_selection_strategy, inplace=True
        )

    for review in base_reviewset:
        for label_id in remove_label_ids:
            review.remove_label(label_id)

    if os.path.exists(args.out_file):
        print(f"\nFile {bcolors.RED}{args.out_file}{bcolors.ENDC} already exists")
        if input("Do you want to overwrite it? [Y/n]: ").lower() == "n":
            print(f"{bcolors.DARKYELLOW}Aborted!{bcolors.ENDC}")
            return

    print(
        f"\nSaving extracted reviewset to {bcolors.BLUE}{args.out_file}{bcolors.ENDC}"
    )
    print_stats(base_reviewset)

    if input("\nDo you want to save the file? [Y/n]: ").lower() == "n":
        print(f"{bcolors.DARKYELLOW}Aborted!{bcolors.ENDC}")
        return
    base_reviewset.save_as(args.out_file)
    print(f"{bcolors.GREEN}Saved!{bcolors.ENDC}")


def delete(base_reviewset: ReviewSet, args: argparse.Namespace):
    delete_label_ids = filter_invalid_labels(base_reviewset, args.label_ids)
    if not delete_label_ids:
        print(f"\n{bcolors.DARKYELLOW}Found no labels to delete.{bcolors.ENDC}")
        return

    if args.force:
        confirm = True
    else:
        print(
            f"\nThe following {bcolors.BLUE}{len(delete_label_ids)}{bcolors.ENDC} label(s) will be deleted from the base file:\n\t",
            f"\n\t".join(
                f"{bcolors.RED}{label}{bcolors.ENDC}" for label in delete_label_ids
            ),
            sep="",
        )
        confirm = not (
            input(
                f"\nDo you really want to delete these label(s) from the base file? [Y/n]: "
            ).lower()
            == "n"
        )

    if confirm:
        for label_id in delete_label_ids:
            for review in base_reviewset:
                review.remove_label(label_id)
        base_reviewset.save()
        print(
            f"\n{bcolors.GREEN}Deleted {len(delete_label_ids)} label(s) from the base reviewset!{bcolors.ENDC}"
        )
    else:
        print(f"\n{bcolors.DARKYELLOW}Aborted!{bcolors.ENDC}")


def sample(base_reviewset: ReviewSet, args: argparse.Namespace):
    if not args.output and args.quiet:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Please specify either an output filepath or don't use the --quiet flag"
        )
        return

    if args.n is None and not args.all:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Please specify either the number of reviews to sample or use the --all flag"
        )
        return

    if args.n and args.n > len(base_reviewset):
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Cannot sample {bcolors.RED}{args.n}{bcolors.ENDC} reviews from a reviewset with only {bcolors.BLUE}{len(base_reviewset)}{bcolors.ENDC} reviews"
        )
        return

    if args.n and args.n < 1:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Cannot sample {bcolors.RED}{args.n}{bcolors.ENDC} reviews, please specify a number greater than 0"
        )
        return

    if args.seed:
        print(f"\nUsing seed {bcolors.BLUE}{args.seed}{bcolors.ENDC} for sampling")

    if args.all:
        cut_reviewset = base_reviewset
    else:
        cut_reviewset, _ = base_reviewset.split(
            (args.n / len(base_reviewset)), seed=args.seed
        )

    if not args.quiet:
        print(f"\nPrinting {bcolors.BLUE}{args.n}{bcolors.ENDC} sampled reviews:")
        for review in cut_reviewset:
            print(review)

    if args.output:
        if os.path.exists(args.output):
            print(f"\nFile {bcolors.RED}{args.output}{bcolors.ENDC} already exists")
            if input("Do you want to overwrite it? [Y/n]: ").lower() == "n":
                print(f"\n{bcolors.DARKYELLOW}Aborted!{bcolors.ENDC}")
                return

        cut_reviewset.save_as(args.output)
        print(
            f"\nSuccessfully saved {bcolors.BLUE}{args.n}{bcolors.ENDC} sampled reviews them to {bcolors.BLUE}{args.output}{bcolors.ENDC}"
        )


def annotate(base_reviewset: ReviewSet, args: argparse.Namespace):
    from training.generator import Generator, DEFAULT_GENERATION_CONFIG

    label_id = None
    if args.last_part_of_label_id is not None:
        label_id = f"model-{args.artifact_name}-{args.last_part_of_label_id}"

    if label_id and label_id in base_reviewset.get_all_label_ids():
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Label {bcolors.DARKYELLOW}{label_id}{bcolors.ENDC} already exists in the reviewset"
        )
        return

    generator = Generator(
        args.artifact_name,
        args.generation_config or DEFAULT_GENERATION_CONFIG,
        args.checkpoint,
    )

    generator.generate_label(base_reviewset, label_id=label_id, verbose=not args.quiet)

    if label_id:
        base_reviewset.save()


def score(base_reviewset: ReviewSet, args: argparse.Namespace):
    all_label_ids = base_reviewset.get_all_label_ids()

    reference_label_ids = filter_invalid_labels(
        base_reviewset, args.reference_label_candidate_ids.split(",")
    )
    if not reference_label_ids:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} No reference labels found in the base reviewset"
        )
        return

    label_ids = []
    if args.all:
        label_ids = list(all_label_ids - set(reference_label_ids))
    elif args.label_ids:
        label_ids = filter_invalid_labels(base_reviewset, args.label_ids)
    else:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Either --all or at least one label must be specified for scoring"
        )
        return
    if not label_ids:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} No labels to score found in the base reviewset"
        )
        return

    scores = {}
    for label_id in label_ids:
        if args.metrics is None:
            scores[label_id] = base_reviewset.get_agg_scores(
                label_id,
                *reference_label_ids,
            )
        else:
            scores[label_id] = base_reviewset.get_agg_scores(
                label_id, *reference_label_ids, list(args.metrics.split(","))
            )

    print(
        f"\nScores against the reference label (candidates): ",
        ", ".join(
            f"{bcolors.BLUE}{ref_id}{bcolors.ENDC}" for ref_id in reference_label_ids
        )
        + "\n\t",
        "\n\t".join(
            [
                f"{bcolors.BLUE}{label_id}{bcolors.ENDC}: {pprint.pformat(score)}"
                for label_id, score in scores.items()
            ]
        ),
        sep="",
    )
    base_reviewset.save()
    print(f"\n{bcolors.GREEN}Scores saved!{bcolors.ENDC}")


def remove_outliers(base_reviewset: ReviewSet, args: argparse.Namespace):
    from clustering import utils
    from clustering.clusterer import Clusterer
    from clustering.data_loader import DataLoader
    import pandas as pd
    from copy import deepcopy

    clustering_config = {
        "data": {
            "model_name": "all-mpnet-base-v2",
            "dim_reduction": "tsne",
            "n_components": 2,
        },
        "clustering": {
            "use_reduced_embeddings": False,
            "algorithm": "agglomerative",
            "metric": "cosine",
            "linkage": "average",
            "save_to_disk": False,
            "distance_thresholds": [args.dist_threshold],
        },
    }

    arg_dicts = utils.get_arg_dicts(clustering_config, len(base_reviewset))
    review_set_df = DataLoader(
        [args.base_file], args.label_id, clustering_config["data"]
    ).load()
    clustered_df = Clusterer(review_set_df, arg_dicts[0]).cluster()

    # remove outliers

    outlier_df = pd.DataFrame()
    # add label_id to outlier_df
    total_reviews = len(clustered_df)
    print(f"Removing outliers from {total_reviews} reviews")
    count_label_df = (
        clustered_df.groupby("label")
        .count()
        .reset_index()
        .sort_values(ascending=True, by="review_id")
    )
    for label in count_label_df["label"]:
        print(label)
        if outlier_df.shape[0] / total_reviews > 1 - args.keep:
            break
        # add all reviews of label to outlier_df
        outlier_df = outlier_df.append(clustered_df[clustered_df["label"] == label])
    print(f"Removing {outlier_df.shape[0]} reviews as outliers")

    new_reviewset = deepcopy(base_reviewset)
    new_reviewset.remove_outliers(
        outlier_df[["review_id", "usage_option"]], args.label_id
    )

    new_reviewset.save_as(args.output_file)


def main():
    args, help_text = parse_args()

    base_reviewset = ReviewSet.from_files(args.base_file)
    print(
        f"Using file: {bcolors.BLUE}{base_reviewset.save_path}{bcolors.ENDC} as base reviewset"
    )
    print_stats(base_reviewset)

    command = args.command
    if command:
        eval(f"{command}(base_reviewset, args)")


if __name__ == "__main__":
    main()
