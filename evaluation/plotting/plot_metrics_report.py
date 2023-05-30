import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_report(df: pd.DataFrame, metrics) -> list[sns.barplot]:
    plots = []
    for metric in metrics:
        plt.clf()
        aggregations = ["count", "mean", "std"]
        total_score = df.agg({metric: aggregations})
        usage_options_score = df.groupby("has_usage_options").agg(
            {metric: aggregations}
        )
        group_scores = df.groupby(["star_rating", "has_usage_options"]).agg(
            {metric: aggregations}
        )
        plot = sns.barplot(
            df,
            x="star_rating",
            hue="has_usage_options",
            y=metric,
            errorbar="se",
        )
        plot.set(
            title=f"{total_score}\n\n\n {usage_options_score}\n\n {group_scores}\n\n"
        )
        plots.append(plot)

    return plots
