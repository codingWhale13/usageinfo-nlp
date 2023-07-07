import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import expected_max_performance


def main():
    # The N validation accuracies can go here:
    validation_weighted_mean_f1 = [
        0.742,
        0.7353,
        0.7548,
        0.7294,
        0.7389,
        0.7403,
        0.748,
        0.7376,
        0.7358,
        0.7439,
        0.7449,
        0.7309,
        0.7418,
        0.7331,
        0.7466,
        0.7435,
        0.7492,
        0.7406,
        0.7437,
        0.748,
        0.7474,
        0.7369,
        0.741,
        0.7535,
        0.7445,
        0.7467,
        0.7533,
        0.7439,
        0.745,
        0.7393,
        0.7438,
        0.7437,
        0.7458,
        0.7468,
        0.7419,
    ]
    validation_classification = []

    assert len(validation_weighted_mean_f1) == 35

    data_weighted_mean_f1 = expected_max_performance.samplemax(
        validation_weighted_mean_f1
    )
    data_weighted_mean_f1 = expected_max_performance.samplemax(
        validation_classification
    )

    # shading +/- the standard error (similar to standard deviation).
    one_plot(
        data,
        "FLAN-T5-Base",
        logx=False,
        plot_errorbar=True,
        avg_time=0,
        performance_metric="weighted_mean_f1",
    )
    one_plot(
        data,
        "FLAN-T5-Base",
        logx=False,
        plot_errorbar=True,
        avg_time=0,
        performance_metric="correct_classification_fraction",
    )


def one_plot(
    data,
    data_name,
    logx=False,
    plot_errorbar=True,
    avg_time=0,
    performance_metric="accuracy",
):
    # to set default values
    linestyle = "-"
    linewidth = 3
    errorbar_kind = "shade"
    errorbar_alpha = 0.1
    fontsize = 16
    x_axis_time = avg_time != 0

    _, cur_ax = plt.subplots(1, 1)
    cur_ax.set_title(data_name, fontsize=fontsize)
    cur_ax.set_ylabel("Expected validation " + performance_metric, fontsize=fontsize)

    if x_axis_time:
        cur_ax.set_xlabel("Training duration", fontsize=fontsize)
    else:
        cur_ax.set_xlabel("Hyperparameter assignments", fontsize=fontsize)

    if logx:
        cur_ax.set_xscale("log")

    means = data["mean"]
    vars = data["var"]
    max_acc = data["max"]
    min_acc = data["min"]

    if x_axis_time:
        x_axis = [avg_time * (i + 1) for i in range(len(means))]
    else:
        x_axis = [i + 1 for i in range(len(means))]

    if plot_errorbar:
        if errorbar_kind == "shade":
            minus_vars = [
                x - y if (x - y) >= min_acc else min_acc for x, y in zip(means, vars)
            ]
            plus_vars = [
                x + y if (x + y) <= max_acc else max_acc for x, y in zip(means, vars)
            ]
            plt.fill_between(x_axis, minus_vars, plus_vars, alpha=errorbar_alpha)
        else:
            cur_ax.errorbar(
                x_axis, means, yerr=vars, linestyle=linestyle, linewidth=linewidth
            )
    cur_ax.plot(x_axis, means, linestyle=linestyle, linewidth=linewidth)

    left, right = cur_ax.get_xlim()

    plt.xlim((left, right))
    plt.locator_params(axis="y", nbins=10)
    plt.tight_layout()

    save_plot(data_name, logx, plot_errorbar, avg_time)


def save_plot(data_name, logx, plot_errorbar, avg_time):
    name = f"{os.path.dirname(os.path.realpath(__file__))}/plots/{data_name}_logx={logx}_errorbar={plot_errorbar}_avgtime={avg_time}.png"

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(name, dpi=300)


if __name__ == "__main__":
    main()
