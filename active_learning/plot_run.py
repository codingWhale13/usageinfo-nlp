import argparse
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from training.utils import get_model_dir_file_path
import matplotlib.pyplot as plt
from helpers.review_set import ReviewSet
from helpers.label_selection import DatasetSelectionStrategy
from active_learning.dataset_analysis import analyze_review_set
from active_learning.text_entropy import text_entropy
import os
import ast


def absolute_path_from_this_file_to(relative_path: str):
    import os

    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, relative_path)


TRAINING_REVIEW_SET = ReviewSet.from_files(
    absolute_path_from_this_file_to("../ba-30k-train-reviews.json")
)
TRAINING_SELECTION_STRATEGY = DatasetSelectionStrategy("ba-30k-train")
MAX_ACTIVE_LEARNING_ITERATIONS = 40

EASY_ENTROPY_LEVEL = 1.0  # bit

global run_names_report_dir
run_names_report_dir = None
X_START = 0
X_END = 1056

DPI = 300


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


def get_run_name_group(run_name: str) -> str:
    return run_name.split("_run_")[0]


def run_name_groups(run_names: list[str]) -> list[str]:
    return list(set([get_run_name_group(run_name) for run_name in run_names]))


def read_dataframes(run_names: list[str], file_name: str) -> pd.DataFrame:
    dataframes = []
    for run_name in run_names:
        csv_file_path = get_model_dir_file_path(
            run_name + "-active_learning_dir", file_name
        )
        if os.path.isfile(csv_file_path):
            df = pd.read_csv(csv_file_path)
            run_name_group = get_run_name_group(run_name)
            df["run_name"] = run_name
            df["run_name_group"] = run_name_group
            dataframes.append(df)
        else:
            print(csv_file_path)
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
                df.loc[
                    df["run_name"] == run_name, "run_name_group"
                ] = get_run_name_group(run_name)
        all_iterations_df.append(df)
    return pd.concat(all_iterations_df)


def plot_run_scores(run_names: list[str]) -> None:
    all_runs_df = read_dataframes(run_names, "scores_new_3.csv")
    baselines = [
        "random_baseline_100_run_03",
        "random_baseline_050_run_01",
        "random_baseline_050_run_02",
    ]
    baseline_df = read_dataframes(baselines, "scores_new_2.csv")
    for col in [
        "positives_custom_weighted_mean_f1",
        "negatives_custom_weighted_mean_f1",
        "true_positives_custom_weighted_mean_f1",
    ]:
        all_runs_df[f"{col}_mean"] = all_runs_df[col].apply(
            lambda x: ast.literal_eval(x)["mean"]
        )
        baseline_df[f"{col}_mean"] = baseline_df[col].apply(
            lambda x: ast.literal_eval(x)["mean"]
        )
    print(all_runs_df.columns)
    # print(df["positives_custom_weighted_mean_f1"])
    for metric in [
        "test_loss_epoch",
        "harmonic_balanced",
        "custom_weighted_mean_f1_mean",
        "classification_f1",
        "classification_accuracy",
        "positives_custom_weighted_mean_f1_mean",
        "negatives_custom_weighted_mean_f1_mean",
        "true_positives_custom_weighted_mean_f1_mean",
    ]:
        print(metric)
        plt.clf()
        plt.xlim(X_START, X_END)

        plt.grid(True)
        for base_line_group, color in zip(run_name_groups(baselines), ["r", "b"]):
            value = baseline_df[baseline_df["run_name_group"] == base_line_group][
                metric
            ].mean()
            plt.axhline(y=value, color=color, label=base_line_group)

        sns.lineplot(
            data=all_runs_df,
            x="acquired_training_reviews",
            y=metric,
            hue="run_name_group",
            markers=True,
            errorbar="sd",
            style="run_name_group",
        )

        # plt.hlines(y=value, label=f"{base_line_group}", xmin=0, xmax=500)
        plt.savefig(f"{run_names_report_dir}/{metric}.png", dpi=DPI)


def plot_run_datasets_input_word_count(run_names: list[str]) -> None:
    all_iterations_df = read_data_frames_over_all_iteration(
        run_names,
        "training_dataset_iteration",
        count_acquired_training_reviews=True,
    )
    # for iteration in range(0, MAX_ACTIVE_LEARNING_ITERATIONS):
    #     df = read_dataframes(run_names, f"training_dataset_iteration_{iteration}.csv")
    #     if df is None:
    #         continue
    #     for run_name in run_names:
    #         df.loc[df["run_name"] == run_name, "acquired_training_reviews"] = len(
    #             df[df["run_name"] == run_name]
    #         )  # ["acquired_training_reviews"] =
    #     all_iterations_df.append(df)
    # all_iterations_df = pd.concat(all_iterations_df)
    print(all_iterations_df)

    training_dataset_label = "Training dataset average"
    training_dataset_df = get_training_dataset_df()
    mean_input_word_count = training_dataset_df["input_word_count"].mean()
    plt.clf()
    plt.axhline(y=mean_input_word_count, label=training_dataset_label)
    sns.lineplot(
        data=all_iterations_df,
        x="acquired_training_reviews",
        y="input_word_count",
        hue="run_name_group",
        markers=True,
        dashes=False,
        errorbar="sd",
        style="run_name_group",
    )
    plt.xlim(X_START, X_END)

    plt.savefig(f"{run_names_report_dir}/input_word_count.png", dpi=DPI)

    mean_usage_options_word_count = training_dataset_df[
        training_dataset_df["has_usage_options"] == True
    ]["usage_options_word_count"].mean()
    plt.clf()
    plt.axhline(y=mean_usage_options_word_count, label=training_dataset_label)
    sns.lineplot(
        data=all_iterations_df[all_iterations_df["has_usage_options"] == True],
        x="acquired_training_reviews",
        y="usage_options_word_count",
        hue="run_name_group",
        markers=True,
        dashes=False,
        errorbar="sd",
        style="run_name_group",
    )
    plt.xlim(X_START, X_END)

    plt.savefig(f"{run_names_report_dir}/usage_options_word_count.png", dpi=DPI)

    mean_usage_optons_count = training_dataset_df[
        training_dataset_df["has_usage_options"] == True
    ]["usage_options_count"].mean()
    plt.clf()
    plt.axhline(y=mean_usage_optons_count, label=training_dataset_label)
    sns.lineplot(
        data=all_iterations_df[all_iterations_df["has_usage_options"] == True],
        x="acquired_training_reviews",
        y="usage_options_count",
        hue="run_name_group",
        markers=True,
        dashes=False,
        errorbar="sd",
        style="run_name_group",
    )
    plt.xlim(X_START, X_END)

    plt.savefig(f"{run_names_report_dir}/usage_option_count.png", dpi=DPI)


def plot_run_acquisition_function_score(run_names: list[str]) -> None:
    df = read_data_frames_over_all_iteration(run_names, "metric_scores_iteration")
    df = df[~df["entropy"].isnull()]
    print(df)
    has_usage_options_column = []
    star_rating_column = []
    product_category_column = []
    for _, row in df.iterrows():
        review_id = row["review_id"]
        review = TRAINING_REVIEW_SET.get_review(review_id)
        has_usage_options = review.label_has_usage_options(TRAINING_SELECTION_STRATEGY)
        has_usage_options_column.append(has_usage_options)
        star_rating_column.append(int(review.data["star_rating"]))
        product_category_column.append(review.data["product_category"])
    df["has_usage_options"] = has_usage_options_column
    df["star_rating"] = star_rating_column
    df["product_category"] = product_category_column
    print(df)

    for run_name_group in run_name_groups(run_names):
        group_df = df[df["run_name_group"] == run_name_group]
        if len(group_df) == 0:
            continue

        individual_report_dir = run_names_report_dir + "/" + run_name_group
        os.makedirs(individual_report_dir, exist_ok=True)
        possible_iterations = group_df["iteration"].unique()

        # plt.clf()
        # sns.barplot(
        #    data=iteration_group_df, x="entropy", hue=group_attribute
        # )  # multiple="stack")
        # plt.savefig(
        #    f"{individual_report_dir}/entropy_histogram_{group_attribute}_iteration_{iteration}.png"
        # )

        for group_attribute in ["has_usage_options", "star_rating"]:
            plt.clf()
            plot = so.Plot(
                group_df[group_df["entropy"] <= EASY_ENTROPY_LEVEL],
                x="iteration",
                color=group_attribute,
            ).add(so.Bar(), so.Count())
            plot.save(
                f"{run_names_report_dir}/easy_reviews_{group_attribute}_{run_name_group}.png"
            )

        for iteration in possible_iterations:
            iteration_group_df = group_df[group_df["iteration"] == iteration]
            for group_attribute in [
                "has_usage_options",
                "star_rating",
                # ("has_usage_options", "star_rating"),
            ]:
                plt.clf()
                sns.histplot(
                    data=iteration_group_df, x="entropy", hue=group_attribute
                )  # multiple="stack")
                plt.savefig(
                    f"{individual_report_dir}/entropy_histogram_{group_attribute}_iteration_{iteration}.png",
                    dpi=DPI,
                )
            """
            plt.clf()
            g = sns.FacetGrid(iteration_group_df, row="product_category")
            g.map_dataframe(sns.histplot, x="entropy")
            plt.savefig(
                f"{individual_report_dir}/entropy_histogram_product_category_iteration_{iteration}.png"
            )
            plt.close()
            """
    plt.clf()
    sns.lineplot(
        data=df,
        x="iteration",
        y="entropy",
        hue="run_name",
        markers=True,
        style="run_name",
    )
    # plt.xlim(X_START, X_END)
    plt.savefig(f"{run_names_report_dir}/mean_entropy.png", dpi=DPI)
    plt.clf()
    sns.lineplot(
        data=df,
        x="iteration",
        y="entropy",
        hue="has_usage_options",
        markers=True,
        style="has_usage_options",
    )
    # plt.xlim(X_START, X_END)
    plt.savefig(f"{run_names_report_dir}/mean_entropy_per_class.png", dpi=DPI)

    plt.clf()
    sns.lineplot(
        data=df,
        x="iteration",
        y="entropy",
        hue="star_rating",
        # style="star_rating",
    )
    # plt.xlim(X_START, X_END)
    plt.savefig(f"{run_names_report_dir}/mean_entropy_per_star_rating.png", dpi=DPI)


def plot_run_datasets_usage_option_percentage(run_names: list[str]) -> None:
    all_iterations_df = []
    for iteration in range(0, MAX_ACTIVE_LEARNING_ITERATIONS):
        df = read_dataframes(run_names, f"training_dataset_iteration_{iteration}.csv")
        if df is None:
            continue
        frac = (
            df.groupby(["run_name", "run_name_group"])["has_usage_options"]
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
    plt.axhline(
        y=usage_options_percentage, label="Training dataset usage options percentage"
    )
    sns.lineplot(
        data=all_iterations_df,
        x="acquired_training_reviews",
        y="percentage_has_usage_options_True",
        hue="run_name_group",
        markers=True,
        errorbar="sd",
        dashes=False,
        style="run_name_group",
    )
    plt.xlim(left=X_START)

    plt.savefig(
        f"{run_names_report_dir}/percentage_has_usage_options.png",
        dpi=DPI,
    )


def plot_run_dataset_entropy(run_names: list[str]) -> None:
    entropy_df = []
    total_training_text_entropy = text_entropy(
        [
            review.get_prompt(prompt_id="active_learning_v1")
            for review in TRAINING_REVIEW_SET
        ]
    )
    score_df = read_dataframes(run_names, "scores_new_3.csv")
    test_loss_df = score_df#[["test_loss_epoch", "iteration", "acquired_training_reviews"]]

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
                    "run_name_group": get_run_name_group(run_name),
                    "run_name": run_name,
                    "text_entropy": entropy,
                    "acquired_training_reviews": len(review_ids),
                }
            )
    entropy_df = pd.DataFrame.from_records(entropy_df)
    print(entropy_df.columns)
    print(test_loss_df.columns)
    joined_df = pd.merge(entropy_df, test_loss_df, on=["run_name", "run_name_group", "acquired_training_reviews"])
    print(joined_df)
    plt.clf()
    sns.scatterplot(joined_df, x="text_entropy", y="test_loss_epoch", hue="run_name_group")
    plt.savefig(f"{run_names_report_dir}/text_entropy_test_loss_scatter.png", dpi=DPI)
    exit()

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
    plt.xlim(X_START, X_END)
    plt.axhline(y=total_training_text_entropy)
    plt.savefig(f"{run_names_report_dir}/text_entropy.png", dpi=DPI)


if __name__ == "__main__":
    args = parse_args()

    run_names_report_dir = absolute_path_from_this_file_to(
        f"score_final_final/{'_'.join(args.run_names)}"
    )
    os.makedirs(run_names_report_dir, exist_ok=True)

    plot_run_dataset_entropy(args.run_names)
    exit()
    plot_run_datasets_input_word_count(args.run_names)
    plot_run_datasets_usage_option_percentage(args.run_names)
    plot_run_scores(args.run_names)
    plot_run_dataset_entropy(args.run_names)
    plot_run_acquisition_function_score(args.run_names)
