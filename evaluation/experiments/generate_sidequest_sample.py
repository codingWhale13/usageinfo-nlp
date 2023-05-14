from helpers.review_set import ReviewSet
import random
from evaluation.scoring.core import get_similarity, get_all_similarities
import itertools
import argparse
import pickle
from evaluation.scoring.evaluation_cache import EvaluationCache


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Generate a sample of usage option pairs to evaluate different similarity metrics."
    )
    parser.add_argument(
        "-r",
        "--review_set",
        type=str,
        required=True,
        help="Path to review set to use for creating usage options pairs. They will be randomly sampled from the usage options in the review set.",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=35,
        help="Number of usage option pairs to generate from review_set.",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=list,
        default=["chat_gpt*"],
        nargs="*",
        help="Label id of the usage options to use for generating the pairs. Default is chat_gpt*, which will use any usage option with a label id starting with chat_gpt.",
    )
    parser.add_argument(
        "-g",
        "--golden",
        type=str,
        required=True,
        help="Path to golden review set to use for creating usage options pairs. They will be randomly sampled from the usage options in the review set.",
    )
    parser.add_argument(
        "-lg",
        "--label_golden",
        type=str,
        default="bp-golden_v3",
        help="Label id of the usage options to use for generating the pairs from golden reviews. Default is bp-golden_v3, which will use any usage option with a label id starting with bp-golden_v3.",
    )
    parser.add_argument(
        "-lgc",
        "--label_golden_comparator",
        type=str,
        required=True,
        nargs="+",
        help="Label id of the usage options to use as a comparison for the golden usage options.",
    )
    parser.add_argument(
        "-ng",
        "--number_golden",
        type=int,
        default=85,
        help="Number of usage option pairs to generate from golden.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output file to save the generated usage option pairs. It will be a pickle file.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
        help="Seed to use for random sampling.",
    )
    return parser.parse_known_args()


def main():
    def pop_random(lst):
        idx = random.randrange(0, len(lst))
        return lst.pop(idx)

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    args, _ = arg_parse()
    random.seed(args.seed)

    reviews = ReviewSet.from_files(args.review_set)

    usage_options = f7(
        [
            usage_option.lower()
            for usage_option in reviews.get_usage_options(*args.label)
        ]
    )

    pairs = []

    for i in range(args.number):
        rand1 = pop_random(usage_options)
        rand2 = pop_random(usage_options)
        pair = rand1, rand2, "non_golden"
        pairs.append(pair)

    reviews2 = ReviewSet.from_files(args.golden)

    pairs2 = []

    for review in reviews2:
        usage_options_golden = f7(
            [
                usage_option.lower()
                for usage_option in review.get_usage_options(args.label_golden)
            ]
        )
        usage_options = f7(
            [
                usage_option.lower()
                for usage_option in review.get_usage_options(
                    *args.label_golden_comparator
                )
            ]
        )

        for i in range(len(usage_options_golden)):
            for j in range(len(usage_options)):
                pairs2.append((usage_options_golden[i], usage_options[j], "golden"))

    random.shuffle(pairs2)

    pairs = pairs[: args.number]
    pairs2 = pairs2[: args.number_golden]

    pairs = pairs + pairs2
    random.shuffle(pairs)

    print(pairs)

    pairs = {
        pair: {
            "similarities": {
                comparator: value
                for comparator, value in get_all_similarities(pair[0], pair[1])
            }
        }
        for pair in pairs
    }

    # print(pairs)

    import json

    print(pairs)
    with open(args.output, "wb") as f:
        pickle.dump(pairs, f)

    EvaluationCache.get().save_to_disk()


if __name__ == "__main__":
    main()
