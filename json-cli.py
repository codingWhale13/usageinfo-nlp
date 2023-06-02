#!/usr/bin/env python3
import argparse
import os
import copy
import pprint

from helpers.review_set import ReviewSet
from training.generator import DEFAULT_GENERATION_CONFIG

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
    parser.add_argument(
        "base_file",
        type=str,
        help="Filepath of the base reviewset that you want to use",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge reviews and/or labels of other files into the base reviewset",
    )
    merge_parser.add_argument(
        "merge_files",
        type=str,
        nargs="+",
        metavar="merge_file",
        help="Reviewset file to with labels to merge into the base file",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract one or more label(s) from the base reviewset to a new file",
    )
    extract_parser.add_argument(
        "labels",
        type=str,
        nargs="+",
        metavar="label",
        help="Label(s) to extract from the base file",
    )
    extract_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        metavar="fileapath",
        help="Filepath to save the extracted reviews to",
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
        "labels",
        type=str,
        nargs="+",
        metavar="label",
        help="Label(s) to delete from the base file",
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
        "n",
        type=int,
        nargs="?",
        help="Number of reviews to sample from the base reviewset",
    )
    sample_parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="fileapath",
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
        "--seed", "-s", type=int, default=None, help="Seed to use for sampling"
    )

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotate the base reviewset with a trained model artifact",
    )
    annotate_parser.add_argument(
        "artifact_name",
        type=str,
        metavar="model_artifact",
        help="Model artifact to use for annotation",
    )
    annotate_parser.add_argument(
        "label_id",
        type=str,
        nargs="?",
        default=None,
        help="ID of the label that the annotation should be saved under",
    )
    annotate_parser.add_argument(
        "--checkpoint",
        "-c",
        type=int,
        help="Optional checkpoint of the artifact to use for annotation",
    )
    annotate_parser.add_argument(
        "--generation_config",
        "-g",
        type=str,
        default=DEFAULT_GENERATION_CONFIG,
        help="Generation config to use for the annotation",
    )
    annotate_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output of the annotation"
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score labels against each other",
    )
    score_parser.add_argument(
        "label",
        type=str,
        help="Label to score",
    )
    score_parser.add_argument(
        "reference_labels",
        nargs="*",
        type=str,
        help="Reference labels to score against",
    )
    score_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Score the label against all other labels",
    )
    score_parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        help="Metrics to use for scoring",
    )

    return parser.parse_args(), parser.format_help()


def print_stats(reviewset: ReviewSet, prefix: str = ""):
    labels = list(reviewset.get_all_label_ids())
    labels_with_count = [
        (label, len(reviewset.reviews_with_labels({label}))) for label in labels
    ]

    print(
        f"{prefix}The reviewset contains {bcolors.BLUE}{len(reviewset)}{bcolors.ENDC} reviews and the following {bcolors.BLUE}{len(reviewset.get_all_label_ids())}{bcolors.ENDC} label(s):\n{prefix}\t",
        dash + f"\n{prefix}\t",
        "{1:<40}{2:<40}\n{0}\t".format(prefix, "Label", "Coverage"),
        dash + f"\n{prefix}\t",
        f"\n{prefix}\t".join(
            f"{label:<40}{str(count) + '/' + str(len(reviewset)):<40}"
            for label, count in labels_with_count
        ),
        sep="",
    )


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

        new_labels = reviewset.get_all_label_ids() - base_reviewset.get_all_label_ids()
        new_labels_with_base_count = []
        for label in new_labels:
            filtered_reviewset = reviewset.filter_with_label_strategy(
                LabelIDSelectionStrategy(label), inplace=False
            )
            new_labels_with_base_count.append(
                (label, base_reviewset.count_common_reviews(filtered_reviewset))
            )

        if not new_labels:
            print(f"\n\t{bcolors.DARKYELLOW}No new labels found.{bcolors.ENDC}")
            if not allow_new_reviews or new_review_count == 0:
                print(f"\n\t{bcolors.DARKYELLOW}Skipping...{bcolors.ENDC}")
                continue
        else:
            print(
                f"\n\tFor the following {bcolors.BLUE}{len(new_labels)}{bcolors.ENDC} new label(s), that many reviews labelled by them could also be found in the base reviewset:\n",
                "\t\t" + dash + "\n\t\t",
                "{0:<40}{1:<40}\n\t\t".format(
                    "Label", "Coverage of reviews in base file"
                ),
                dash + f"\n\t\t",
                f"\n\t\t".join(
                    f"{label:<40}{str(count) + '/' + str(len(base_reviewset)):<40}"
                    for label, count in new_labels_with_base_count
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

    extract_labels = copy.copy(args.labels)
    all_labels = base_reviewset.get_all_label_ids()
    for label in args.labels:
        if label not in all_labels:
            print(
                f"\nLabel {bcolors.DARKYELLOW}{label}{bcolors.ENDC} does not exist, skipping..."
            )
            extract_labels.remove(label)

    if not extract_labels:
        print(f"\n{bcolors.DARKYELLOW}No labels to extract, aborting...{bcolors.ENDC}")
        return

    remove_labels = list(all_labels - set(extract_labels))

    if not args.keep:
        label_selection_strategy = LabelIDSelectionStrategy(*extract_labels)
        base_reviewset.filter_with_label_strategy(
            label_selection_strategy, inplace=True
        )

    for review in base_reviewset:
        for label in remove_labels:
            review.remove_label(label)

    if os.path.exists(args.output):
        print(f"\nFile {bcolors.RED}{args.output}{bcolors.ENDC} already exists")
        if input("Do you want to overwrite it? [Y/n]: ").lower() == "n":
            print(f"{bcolors.DARKYELLOW}Aborted!{bcolors.ENDC}")
            return

    print(f"\nSaving extracted reviewset to {bcolors.BLUE}{args.output}{bcolors.ENDC}")
    base_reviewset.save_as(args.output)
    print(f"\n{bcolors.GREEN}Saved!{bcolors.ENDC}")


def delete(base_reviewset: ReviewSet, args: argparse.Namespace):
    delete_labels = copy.copy(args.labels)
    for label in args.labels:
        if label not in base_reviewset.get_all_label_ids():
            print(
                f"\nLabel {bcolors.DARKYELLOW}{label}{bcolors.ENDC} does not exist, skipping..."
            )
            delete_labels.remove(label)

    if not delete_labels:
        print(f"\n{bcolors.DARKYELLOW}Found no labels to delete.{bcolors.ENDC}")
        return

    if args.force:
        confirm = True
    else:
        print(
            f"\nThe following {bcolors.BLUE}{len(delete_labels)}{bcolors.ENDC} label(s) will be deleted from the base file:\n\t",
            f"\n\t".join(
                f"{bcolors.RED}{label}{bcolors.ENDC}" for label in delete_labels
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
        for label in delete_labels:
            for review in base_reviewset:
                review.remove_label(label)
        base_reviewset.save()
        print(
            f"\n{bcolors.GREEN}Deleted {len(delete_labels)} label(s) from the base reviewset!{bcolors.ENDC}"
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
    from training.generator import Generator
    from training import utils

    label_id = None
    if args.label_id is not None:
        label_id = f"model-{args.artifact_name}-{args.label_id}"

    if label_id and label_id in base_reviewset.get_all_label_ids():
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Label {bcolors.DARKYELLOW}{label_id}{bcolors.ENDC} already exists in the reviewset"
        )
        return

    generator = Generator(args.artifact_name, args.generation_config, args.checkpoint)

    generator.generate_label(base_reviewset, label_id=label_id, verbose=not args.quiet)

    if label_id:
        base_reviewset.save()


def score(base_reviewset: ReviewSet, args: argparse.Namespace):
    if not args.all and not args.reference_labels:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Either --all or at least one reference label must be specified for scoring"
        )
        return

    all_labels = list(base_reviewset.get_all_label_ids())

    if args.label not in all_labels:
        print(
            f"\n{bcolors.RED}Error:{bcolors.ENDC} Label {bcolors.DARKYELLOW}{args.label}{bcolors.ENDC} does not exist in the base reviewset"
        )
        return

    if args.all:
        reference_labels = all_labels
    else:
        reference_labels = copy.copy(args.reference_labels)

        for label in args.reference_labels:
            if label not in all_labels:
                print(
                    f"\nLabel {bcolors.DARKYELLOW}{label}{bcolors.ENDC} does not exist, skipping..."
                )
                reference_labels.remove(label)

        if len(reference_labels) == 0:
            print(
                f"\n{bcolors.DARKYELLOW}No reference labels left, skipping scoring...{bcolors.ENDC}"
            )
            return

    scores = {}
    for reference_label in reference_labels:
        if args.metrics is None:
            scores[reference_label] = base_reviewset.get_agg_scores(
                args.label,
                reference_label,
            )
        else:
            scores[reference_label] = base_reviewset.get_agg_scores(
                args.label, reference_label, list(args.metrics.split(","))
            )

    for reference_label, score in scores.items():
        print(
            f"\nScores against the label {bcolors.BLUE}{reference_label}{bcolors.ENDC}:"
        )
        pprint.pprint(score)

    base_reviewset.save()
    print(f"\n{bcolors.GREEN}Scores saved!{bcolors.ENDC}")


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
