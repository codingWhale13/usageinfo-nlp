import argparse
import pandas as pd
import seaborn as sns
from training.utils import get_model_dir_file_path
import matplotlib.pyplot as plt
from helpers.review_set import ReviewSet
from helpers.label_selection import DatasetSelectionStrategy
from active_learning.dataset_analysis import analyze_review_set
from active_learning.text_entropy import text_entropy
import os


def absolute_path_from_this_file_to(relative_path: str):
    import os

    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)


TRAINING_REVIEW_SET = ReviewSet.from_files(
    absolute_path_from_this_file_to("../ba-30k-train-reviews.json")
)
TRAINING_SELECTION_STRATEGY = DatasetSelectionStrategy("ba-30k-train")
MAX_ACTIVE_LEARNING_ITERATIONS = 40

global run_names_report_dir
run_names_report_dir = None


def parse_args():
    argparser = argparse.ArgumentParser("Score one or more runs")
    argparser.add_argument(
        "run_names", nargs="+", help="run_names of the runs you want to score"
    )
    return argparser.parse_args()


def training_dataset_df_loader():
    df = None

    def lazy_df():
        nonlocal df
        if df is None:
            df = analyze_review_set(TRAINING_REVIEW_SET, TRAINING_SELECTION_STRATEGY)
        return df

    return lazy_df


get_training_dataset_df = training_dataset_df_loader()


def read_dataframes(run_names: list[str], file_name: str) -> pd.DataFrame:
    dataframes = []
    for run_name in run_names:
        csv_file_path = get_model_dir_file_path(
            run_name + "-active_learning_dir", file_name
        )
        if os.path.isfile(csv_file_path):
            df = pd.read_csv(csv_file_path)
            df["run_name"] = run_name
            dataframes.append(df)
    if len(dataframes) == 0:
        return None
    return pd.concat(dataframes)


def read_data_frames_over_all_iteration(
    run_names: list[str], base_file_name: str, count_acquired_training_reviews=False
) -> pd.DataFrame:
    all_iterations_df = []
    for iteration in range(0, MAX_ACTIVE_LEARNING_ITERATIONS):
        df = read_dataframes(run_names, f"{base_file_name}_{iteration}.csv")
        if df is None:
            continue
        df["iteration"] = iteration
        if count_acquired_training_reviews:
            for run_name in run_names:
                df.loc[df["run_name"] == run_name, "acquired_training_reviews"] = len(
                    df[df["run_name"] == run_name]
                )
        all_iterations_df.append(df)
    return pd.concat(all_iterations_df)


def plot_run_scores(run_names: list[str]) -> None:
    all_runs_df = read_dataframes(run_names, "scores.csv")
    print(all_runs_df)
    for metric in [
        "validation_loss",
        "harmonic_balanced",
    ]:  # "custom_weighted_mean_f1_mean"
        plt.clf()
        sns.lineplot(
            data=all_runs_df,
            x="acquired_training_reviews",
            y=metric,
            hue="run_name",
            markers=True,
            style="run_name",
        )
        plt.savefig(f"{run_names_report_dir}/{metric}-{'_'.join(run_names)}.png")


def plot_run_datasets_input_word_count(run_names: list[str]) -> None:
    all_iterations_df = []
    for iteration in range(0, MAX_ACTIVE_LEARNING_ITERATIONS):
        df = read_dataframes(run_names, f"training_dataset_iteration_{iteration}.csv")
        if df is None:
            continue
        for run_name in run_names:
            df.loc[df["run_name"] == run_name, "acquired_training_reviews"] = len(
                df[df["run_name"] == run_name]
            )  # ["acquired_training_reviews"] =
        all_iterations_df.append(df)
    all_iterations_df = pd.concat(all_iterations_df)
    print(all_iterations_df)

    training_dataset_df = get_training_dataset_df()
    mean_input_word_count = training_dataset_df["input_word_count"].mean()
    plt.clf()
    sns.lineplot(
        data=all_iterations_df,
        x="acquired_training_reviews",
        y="input_word_count",
        hue="run_name",
        markers=True,
        dashes=False,
        style="run_name",
    )
    plt.axhline(y=mean_input_word_count)
    plt.savefig(f"{run_names_report_dir}/input_word_count{'_'.join(run_names)}.png")

    mean_usage_options_word_count = training_dataset_df[
        training_dataset_df["has_usage_options"] == True
    ]["usage_options_word_count"].mean()
    plt.clf()
    sns.lineplot(
        data=all_iterations_df[all_iterations_df["has_usage_options"] == True],
        x="acquired_training_reviews",
        y="usage_options_word_count",
        hue="run_name",
        markers=True,
        dashes=False,
        style="run_name",
    )
    plt.axhline(y=mean_usage_options_word_count)
    plt.savefig(
        f"{run_names_report_dir}/usage_options_word_count{'_'.join(run_names)}.png"
    )

    mean_usage_optons_count = training_dataset_df[
        training_dataset_df["has_usage_options"] == True
    ]["usage_options_count"].mean()
    plt.clf()
    sns.lineplot(
        data=all_iterations_df[all_iterations_df["has_usage_options"] == True],
        x="acquired_training_reviews",
        y="usage_options_count",
        hue="run_name",
        markers=True,
        dashes=False,
        style="run_name",
    )
    plt.axhline(y=mean_usage_optons_count)
    plt.savefig(f"{run_names_report_dir}/usage_option_count{'_'.join(run_names)}.png")


def plot_run_acquisition_function_score(run_names: list[str]) -> None:
    df = read_data_frames_over_all_iteration(run_names, "metric_scores_iteration")
    df = df[~df["entropy"].isnull()]

    plt.clf()
    sns.lineplot(
        data=df,
        x="iteration",
        y="entropy",
        hue="run_name",
        markers=True,
        style="run_name",
    )
    plt.savefig(f"{run_names_report_dir}/mean_entropy_{'_'.join(run_names)}")


def plot_run_datasets_usage_option_percentage(run_names: list[str]) -> None:
    all_iterations_df = []
    for iteration in range(0, MAX_ACTIVE_LEARNING_ITERATIONS):
        df = read_dataframes(run_names, f"training_dataset_iteration_{iteration}.csv")
        if df is None:
            continue
        frac = (
            df.groupby(["run_name"])["has_usage_options"]
            .value_counts(normalize=True)
            .unstack(-1)
            .fillna(0)
            .add_prefix("percentage_has_usage_options_")
        )
        frac["iteration"] = iteration
        frac = frac.reset_index()
        print(frac)
        for run_name in run_names:
            frac.loc[frac["run_name"] == run_name, "acquired_training_reviews"] = len(
                df[df["run_name"] == run_name]
            )  # ["acquired_training_reviews"] =
        all_iterations_df.append(frac)
    all_iterations_df = pd.concat(all_iterations_df)
    print(all_iterations_df)

    training_dataset_df = get_training_dataset_df()
    usage_options_percentage = len(
        training_dataset_df[training_dataset_df["has_usage_options"] == True]
    ) / len(training_dataset_df)
    plt.clf()
    sns.lineplot(
        data=all_iterations_df,
        x="acquired_training_reviews",
        y="percentage_has_usage_options_True",
        hue="run_name",
        markers=True,
        dashes=False,
        style="run_name",
    )
    plt.axhline(y=usage_options_percentage)
    plt.savefig(
        f"{run_names_report_dir}/percentage_has_usage_options{'_'.join(run_names)}.png"
    )


def plot_run_dataset_entropy(run_names: list[str]) -> None:
    entropy_df = []
    total_training_text_entropy = text_entropy(
        [
            review.get_prompt(prompt_id="active_learning_v1")
            for review in TRAINING_REVIEW_SET
        ]
    )
    for iteration in range(0, MAX_ACTIVE_LEARNING_ITERATIONS):
        df = read_dataframes(run_names, f"training_dataset_iteration_{iteration}.csv")
        if df is None:
            continue
        for run_name in run_names:
            review_ids = df[df["run_name"] == run_name]["review_id"].tolist()
            if len(review_ids) == 0:
                continue
            run_name_reviews = TRAINING_REVIEW_SET.filter(
                lambda x: x.review_id in review_ids, inplace=False
            )
            entropy = text_entropy(
                [
                    review.get_prompt(prompt_id="active_learning_v1")
                    for review in run_name_reviews
                ]
            )
            entropy_df.append(
                {
                    "run_name": run_name,
                    "text_entropy": entropy,
                    "acquired_training_reviews": len(review_ids),
                }
            )
    entropy_df = pd.DataFrame.from_records(entropy_df)
    plt.clf()
    sns.lineplot(
        data=entropy_df,
        x="acquired_training_reviews",
        y="text_entropy",
        markers=True,
        dashes=False,
        style="run_name",
        hue="run_name",
    )

    plt.axhline(y=total_training_text_entropy)
    plt.savefig(f"{run_names_report_dir}/text_entropy{'_'.join(run_names)}.png")


if __name__ == "__main__":
    args = parse_args()

    run_names_report_dir = absolute_path_from_this_file_to(
        f"score_reports/{'_'.join(args.run_names)}"
    )
    os.makedirs(run_names_report_dir, exist_ok=True)

    plot_run_dataset_entropy(args.run_names)
    plot_run_scores(args.run_names)
    plot_run_datasets_usage_option_percentage(args.run_names)
    plot_run_datasets_input_word_count(args.run_names)
    plot_run_acquisition_function_score(args.run_names)
