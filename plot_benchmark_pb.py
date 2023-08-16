# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv("ba_results/pb_generator_2/results.csv")
emissions = pd.read_csv("ba_results/pb_generator_2/emissions.csv")
emissions["duration"] = emissions["duration"] / 60
results
# %%
emissions
# %%
top_k_results = results[results["parameter_id"] <= 6]
minimum_probability_results = results[
    (5 < results["parameter_id"]) & (results["parameter_id"] <= 13)
]
length_results = results[14 < results["parameter_id"]]
len(top_k_results), len(minimum_probability_results), len(length_results)


def add_x_column(df: pd.DataFrame):
    x = df["tracking_name"].tolist()
    x_column_name, sample_x_column_value = x[0].split(":")
    if "." in sample_x_column_value:
        conversion_function = float
    else:
        conversion_function = int

    x_column_values = [conversion_function(x_i.split(":")[1]) for x_i in x]
    df[x_column_name] = x_column_values
    return df


top_k_emissions = add_x_column(emissions[:6].copy())
minimum_probability_emissions = add_x_column(emissions[6:13].copy())
length_emissions = add_x_column(emissions[13:].copy())


# %%
def plot_parameter(
    df_results,
    df_emissions,
    parameter_name: str,
    axis_1_y_label: str = None,
    axis_2_y_label: str = None,
    title: str = None,
    reverse_x_axis=False,
    use_x_log_scale=False,
):
    plt.clf()

    fig, ax1 = plt.subplots()

    # Plot the accuracy on the first y-axis (left side)
    sns.lineplot(
        x=parameter_name,
        y="mean_total_probability",
        data=df_results,
        ax=ax1,
        color="b",
        marker="o",
    )
    ax1.tick_params(axis="y", labelcolor="b")

    # Create a second y-axis (right side) for the computation duration
    ax2 = ax1.twinx()
    sns.lineplot(
        x=parameter_name, y="duration", data=df_emissions, ax=ax2, color="r", marker="x"
    )
    if use_x_log_scale:
        plt.xscale("log")
    if axis_2_y_label:
        ax2.set_ylabel(axis_2_y_label)
    if axis_1_y_label:
        ax1.set_ylabel(axis_1_y_label)
    ax2.tick_params(axis="y", labelcolor="r")

    # Add a title
    if title:
        plt.title(title)

    if reverse_x_axis:
        plt.gca().invert_xaxis()

    plt.savefig(f"{parameter_name}-compution.png")
    # Display the plot
    plt.show()


run_time_y_label = "Run time in minutes"
# %%
plot_parameter(
    top_k_results,
    top_k_emissions,
    "token_top_k",
    title="token_top_k: Mean total probability and run time",
    axis_2_y_label=run_time_y_label,
)

# %%
plot_parameter(
    minimum_probability_results,
    minimum_probability_emissions,
    "minimum_probability",
    axis_2_y_label=run_time_y_label,
    reverse_x_axis=True,
    use_x_log_scale=True,
)
print(minimum_probability_emissions)
# %%
plot_parameter(
    length_results,
    length_emissions,
    parameter_name="max_sequence_length",
    axis_2_y_label=run_time_y_label,
)
