import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plotter:
    def plot_usage(self, data, h):
        data["contains_usage"] = data.apply(lambda x: h.label_usage(x), axis=1)
        h.create_barplot(
            data,
            "contains_usage",
            "usage_plot.png",
            "Number of Reviews without/with Usage",
        )

    def plot_positive_usage(self, data, h):
        h.create_barplot(
            data,
            "usage",
            "positive_usage_plot.png",
            "Number of Reviews without/with positive Usage",
        )

    def plot_usage_per_category(self, data, h):
        h.create_histogram(
            data,
            "product_category",
            "usage",
            "usage_plot_per_category.png",
            "Positive usage per category",
            y_label="Number of Reviews",
        )

    def plot_mult_usage_per_category(self, data, h):
        h.create_histogram(
            data,
            "product_category",
            "amount_usage",
            "multiple_usage_plot_per_category.png",
            "Count of positive usage per category",
            y_label="Number of Reviews",
        )

    def plot_usage_per_category_perc(self, data, h):
        data_by_usage = h.create_new_df_with_matching_percentage(
            data, "product_category", "usage", sort=True
        )
        h.create_histogram(
            data_by_usage,
            "product_category",
            "usage",
            "usage_plot_per_category_percentage.png",
            "Positive usage per category in percent",
        )

    def plot_mult_usage_per_category_perc(self, data, h):
        data_by_usage = h.create_new_df_with_matching_percentage(
            data, "product_category", "amount_usage", sort=True
        )
        h.create_histogram(
            data_by_usage,
            "product_category",
            "amount_usage",
            "multiple_usage_plot_per_category_percentage.png",
            "Count of positive usage per category in percent",
        )

    def plot_usage_per_length(self, data, h, changed_reviews):
        data_by_usage = h.preprocess_data_for_length(data, changed_reviews, 200)
        h.create_histogram(
            data_by_usage,
            "review_body",
            "usage",
            "usage_plot_per_review_length.png",
            "Positive usage per review length",
            199,
            "Number of Reviews",
        )
        return True

    def plot_usage_per_length_perc(self, data, h, changed_reviews):
        data_by_usage = h.preprocess_data_for_length(data, changed_reviews, 100)
        data_by_usage = h.create_new_df_with_matching_percentage(
            data_by_usage, "review_body", "usage"
        )
        h.create_histogram(
            data_by_usage,
            "review_body",
            "usage",
            "usage_plot_per_review_length_percentage.png",
            "Positive usage per review length in percent",
            99,
        )
        return True

    def plot_mult_usage_per_length(self, data, h, changed_reviews):
        data_by_usage = h.preprocess_data_for_length(data, changed_reviews, 200)
        h.create_histogram(
            data_by_usage,
            "review_body",
            "amount_usage",
            "multiple_usage_plot_per_review_length.png",
            "Count of positive usage per review length",
            199,
            "Number of Reviews",
        )
        return True

    def plot_mult_usage_per_length_perc(self, data, h, changed_reviews):
        data_by_usage = h.preprocess_data_for_length(data, changed_reviews, 100)
        data_by_usage = h.create_new_df_with_matching_percentage(
            data_by_usage, "review_body", "amount_usage"
        )
        h.create_histogram(
            data_by_usage,
            "review_body",
            "amount_usage",
            "multiple_usage_plot_per_review_length_percentage.png",
            "Count of positive usage per review length in percent",
            99,
        )
        return True

    def plot_mult_usage_per_usage(self, data, h):
        data = data.query("usage == 1")
        h.create_barplot(
            data,
            "amount_usage",
            "multiple_usage_plot_per_usage.png",
            "Number of reviews with x positive usages",
        )
        data = (
            data["amount_usage"]
            .value_counts(normalize=True)
            .rename("percentage")
            .mul(100)
            .reset_index()
        )
        labels = data["index"].astype(str)
        h.create_piechart(
            data,
            "percentage",
            labels,
            "multiple_usage_plot_per_usage_pie.png",
            "Percentage of reviews with x positive usages",
        )

    def create_erwartungswert(self, data):
        e = 0
        for i in range(len(data)):
            usage = data.iloc[i]["amount_usage"]
            e += usage
        e = e / len(data)
        print(e)


class Helper:
    def create_histogram(
        self, data, x_label, hue, filename, title, bins=0, y_label="Percentage"
    ):
        plt.figure(constrained_layout=True, figsize=(15, 5))
        if bins == 0:
            sns.histplot(x=x_label, hue=hue, multiple="stack", data=data)
        else:
            sns.histplot(data=data, x=x_label, hue=hue, multiple="stack", bins=bins)
        plt.xticks(rotation=90)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()
        plt.savefig("/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/plots/" + filename)
        plt.clf()

    def create_barplot(self, data, x_label, filename, title, y_label="Percentage"):
        sns.barplot(
            x=x_label,
            data=data,
            estimator=lambda x: len(x) / len(data) * 100,
            y=x_label,
        )
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig("/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/plots/" + filename)
        plt.clf()

    def create_piechart(self, data, x_label, labels, filename, title):
        plt.figure(figsize=(10, 10))
        plt.pie(data[x_label])
        labels = [f"{l}, {s:0.1f}%" for l, s in zip(labels, data[x_label])]
        plt.legend(labels, loc="best", fontsize=12)
        plt.title(title)
        plt.savefig("/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/plots/" + filename)
        plt.clf()

    def create_new_df_with_matching_percentage(
        self, data, input_category, output_category, sort=False
    ):
        data = (
            data.groupby([input_category])[output_category]
            .value_counts(normalize=True)
            .rename("percentage")
            .mul(100)
            .reset_index()
        )
        if sort:
            data = data.sort_values(by=["percentage"], ascending=False)
        new_data = []
        for cat in data[input_category].unique():
            c = 100
            number_usage = data[data[input_category] == cat]["percentage"].values
            for i in range(len(number_usage)):
                for j in range(int(number_usage[i])):
                    df = pd.DataFrame(
                        {input_category: cat, output_category: i}, index=[c]
                    )
                    new_data.append(df)
                    c -= 1
            for k in range(c):
                df = pd.DataFrame({input_category: cat, output_category: 0}, index=[i])
                new_data.append(df)
        return pd.concat(new_data)

    def preprocess_data_for_length(self, data, changed_reviews, max_length=100):
        if not changed_reviews:
            data["review_body"] = data["review_body"].apply(lambda x: len(x.split()))
        return data.query("review_body < @max_length")

    def label_usage(self, row):
        if row["label"] == "-":
            return False
        else:
            return True

    def positive_usage(self, row):
        if row["label"] != "-" and row["label"].split(",")[2] == "1":
            return True
        else:
            return False

    def mult_usage(self, row):
        if row["label"] != "-":
            usages = row["label"].split(",")
            count = 0
            for i in range(2, len(usages), 3):
                if usages[i] == "1":
                    count += 1
            return count
        else:
            return 0


def plot_all(data, p, h, changed_reviews):
    p.plot_usage(data, h)
    p.plot_positive_usage(data, h)
    p.plot_usage_per_category(data, h)
    p.plot_mult_usage_per_category(data, h)
    p.plot_usage_per_category_perc(data, h)
    p.plot_mult_usage_per_category_perc(data, h)
    changed_reviews = p.plot_usage_per_length(data, h, changed_reviews)
    changed_reviews = p.plot_usage_per_length_perc(data, h, changed_reviews)
    changed_reviews = p.plot_mult_usage_per_length(data, h, changed_reviews)
    changed_reviews = p.plot_mult_usage_per_length_perc(data, h, changed_reviews)
    p.plot_mult_usage_per_usage(data, h)
    p.create_erwartungswert(data)


def main():
    print("Start generating plots")
    changed_reviews = False
    h = Helper()
    p = Plotter()
    files = glob.glob(
        "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/samples_labeled/*.tsv"
    )
    data = pd.concat(
        (pd.read_csv(f, sep="\t", quoting=3) for f in files), ignore_index=True
    )
    data["usage"] = data.apply(lambda x: h.positive_usage(row=x), axis=1)
    data["amount_usage"] = data.apply(lambda x: h.mult_usage(row=x), axis=1)
    plot_all(data, p, h, changed_reviews)
    print("Finished generating plots")


if __name__ == "__main__":
    main()
