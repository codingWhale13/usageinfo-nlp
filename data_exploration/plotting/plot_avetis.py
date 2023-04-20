from data_loading import *
from utils import get_slurm_client
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from dask.dataframe import from_pandas
from matplotlib.ticker import MaxNLocator
import numpy as np


from dask.distributed import progress

client = get_slurm_client(nodes=5, processes=2, memory="512GB")

DATA_DIR = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/data_parquet_pp_filtered"
df = read_data(DATA_DIR)

# customize here where to put data and plots

CSV_DIR = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/stats/data_stat_csv_filtered"
TARGET_DIR = (
    "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/stats/data_stat_plots_filtered"
)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)


# correlation

if not os.path.exists(f"{TARGET_DIR}/corr.png"):
    dfc = df.corr().compute()
    plt.figure(figsize=(10, 10))
    sns.heatmap(dfc, annot=True, vmin=-1, vmax=1, fmt=".2f")
    plt.savefig(f"{TARGET_DIR}/corr.png", bbox_inches="tight")
    plt.clf()

# general

if not os.path.exists(f"{CSV_DIR}/describe.csv"):
    dfc = df.describe().compute()
    dfc.to_csv(f"{CSV_DIR}/describe.csv")

if not os.path.exists(f"{CSV_DIR}/number_of_products_customers.csv"):
    num_products = df["product_id"].nunique().compute()
    num_customers = df["customer_id"].nunique().compute()
    result = pd.DataFrame(
        {"products": num_products, "customers": num_customers}, index=[0]
    )
    result.to_csv(f"{CSV_DIR}/number_of_products_customers.csv", index=False)

# single bar plots

bar_plot_columns_binned = [
    "review_body_usage_count",
    "review_body_word_count",
    "helpful_votes",
    "total_votes",
    "review_body_avg_word_length",
    "review_body_flesch_complexity",
    "review_body_usage_density",
]
bar_plot_columns = ["star_rating", "vine", "verified_purchase"]


def load_group_size_csv(df, columns):
    file_name = f"group_size_{'_'.join(columns)}.csv"

    if os.path.exists(f"{CSV_DIR}/{file_name}"):
        return pd.read_csv(f"{CSV_DIR}/{file_name}", index_col=0, header=0)
    else:
        dfc = df.groupby(columns).size().compute()
        dfc.rename("count", inplace=True)
        dfc.sort_index(inplace=True)
        dfc.to_csv(f"{CSV_DIR}/{file_name}")
        return dfc.to_frame()


def count_bar_plot(df, column, binning=True, bucket_size=20):
    dfc = load_group_size_csv(df, [column])

    if not os.path.exists(f"{TARGET_DIR}/group_size_{column}.png"):
        plt.figure(figsize=(10, 10))
        if binning:
            dfc["group"] = pd.qcut(dfc.index, bucket_size, precision=0)
            # print(dfc.head())
            dfc = dfc.groupby("group").sum()
            sns.barplot(
                data=dfc, x=dfc.index, y=dfc["count"], palette="crest", errorbar=None
            )
        else:
            sns.barplot(
                data=dfc, x=dfc.index, y=dfc["count"], palette="crest", errorbar=None
            )

        plt.xlabel(column)
        plt.ylabel("Number of reviews")
        plt.xticks(rotation=90, fontsize=8)
        plt.yscale("log")
        plt.savefig(f"{TARGET_DIR}/group_size_{column}.png", bbox_inches="tight")
        plt.clf()


for column in bar_plot_columns_binned:
    count_bar_plot(df, column)

for column in bar_plot_columns:
    count_bar_plot(df, column, binning=False)

# customer / product

# number of reviews per customer

if not os.path.exists(f"{CSV_DIR}/reviews_per_customer.csv"):
    dfc = (
        df.groupby(["customer_id"])
        .size()
        .rename("count")
        .reset_index()
        .groupby("count")
        .size()
        .rename("customers_with_count_reviews")
        .to_frame()
        .persist()
    )
    progress(dfc)
    dfc.compute().sort_index().to_csv(f"{CSV_DIR}/reviews_per_customer.csv")

if not os.path.exists(f"{TARGET_DIR}/reviews_per_customer.png"):
    dfc = pd.read_csv(f"{CSV_DIR}/reviews_per_customer.csv", index_col=0, header=0)
    dfc["group"] = pd.qcut(dfc.index, 20, precision=0)
    dfc = dfc.groupby("group").sum()
    plt.figure(figsize=(10, 10))
    sns.barplot(
        data=dfc,
        x=dfc.index,
        y=dfc["customers_with_count_reviews"],
        palette="crest",
        errorbar=None,
    )
    plt.xlabel("reviews_per_customer")
    plt.ylabel("Number of customers")
    plt.xticks(rotation=90, fontsize=8)
    plt.yscale("log")
    plt.savefig(f"{TARGET_DIR}/reviews_per_customer.png", bbox_inches="tight")

# number of reviews per product

if not os.path.exists(f"{CSV_DIR}/reviews_per_product.csv"):
    dfc = (
        df.groupby(["product_id"])
        .size()
        .rename("count")
        .reset_index()
        .groupby("count")
        .size()
        .rename("products_with_count_reviews")
        .to_frame()
        .persist()
    )
    progress(dfc)
    dfc.compute().sort_index().to_csv(f"{CSV_DIR}/reviews_per_product.csv")

if not os.path.exists(f"{TARGET_DIR}/reviews_per_product.png"):
    dfc = pd.read_csv(f"{CSV_DIR}/reviews_per_product.csv", index_col=0, header=0)
    dfc["group"] = pd.qcut(dfc.index, 20, precision=0)
    dfc = dfc.groupby("group").sum()
    plt.figure(figsize=(10, 10))
    sns.barplot(
        data=dfc,
        x=dfc.index,
        y=dfc["products_with_count_reviews"],
        palette="crest",
        errorbar=None,
    )
    plt.xlabel("reviews_per_product")
    plt.ylabel("Number of products")
    plt.xticks(rotation=90, fontsize=8)
    plt.yscale("log")
    plt.savefig(f"{TARGET_DIR}/reviews_per_product.png", bbox_inches="tight")
    plt.clf()

# describe vine reviews

if not os.path.exists(f"{CSV_DIR}/describe_vine_reviews.csv"):
    dfc = df[df["vine"] == 1]
    dfc = dfc.describe().compute()
    dfc.to_csv(f"{CSV_DIR}/describe_vine_reviews.csv")


# describe verified purchases

if not os.path.exists(f"{CSV_DIR}/describe_verified_purchases.csv"):
    dfc = df[df["verified_purchase"] == 1]
    dfc = dfc.describe().compute()
    dfc.to_csv(f"{CSV_DIR}/describe_verified_purchases.csv")

# get number of reviews from customers with more than 1000/5000 reviews

if not os.path.exists(f"{CSV_DIR}/no_reviews_result.csv"):

    dfc = pd.read_csv(
        f"{CSV_DIR}/reviews_per_customer.csv", index_col=0, header=0
    ).reset_index()

    dfc_filtered = dfc[dfc["count"] > 5000]
    sum1 = dfc_filtered["count"].dot(dfc_filtered["customers_with_count_reviews"])

    dfc_filtered = dfc[dfc["count"] > 1000]
    sum2 = dfc_filtered["count"].dot(dfc_filtered["customers_with_count_reviews"])

    dfc_filtered = dfc[dfc["count"] < 2]
    sum3 = dfc_filtered["count"].dot(dfc_filtered["customers_with_count_reviews"])

    result = pd.DataFrame(
        {
            "no_reviews_from_customers_with_over_5000_reviews": sum1,
            "no_reviews_from_customers_with_over_1000_reviews": sum2,
            "no_reviews_from_customers_with_less_than_2_reviews": sum3,
        },
        index=[0],
    )
    result.to_csv(f"{CSV_DIR}/no_reviews_result.csv", index=False)

# print("The number of reviews from customers with more than 5000 reviews:" + str(sum))
# print("The number of reviews from customers with more than 1000 reviews:" + str(sum))

# analyze reviews of customers with more than 1000/5000 reviews and less than 2 reviews

if not os.path.exists(f"{CSV_DIR}/group_size_customer_id"):
    grouped = (
        df.groupby("customer_id")
        .size()
        .rename("count")
        .to_frame()
        .reset_index()
        .persist()
    )
    progress(grouped)
    write_data(grouped, f"{CSV_DIR}/group_size_customer_id")

# todo refactor?

if not os.path.exists(f"{CSV_DIR}/reviews_of_customers_with_over_1000_reviews"):
    grouped = read_data(f"{CSV_DIR}/group_size_customer_id")
    grouped1 = grouped[grouped["count"] > 1000]
    reviews_of_customers_with_over_1000_reviews = (
        df.merge(grouped1, on="customer_id", how="inner")
        .drop("count", axis=1)
        .persist()
    )
    progress(reviews_of_customers_with_over_1000_reviews)
    write_data(
        reviews_of_customers_with_over_1000_reviews,
        f"{CSV_DIR}/reviews_of_customers_with_over_1000_reviews",
    )
    if not os.path.exists(
        f"{CSV_DIR}/describe_reviews_of_customers_with_over_1000_reviews.csv"
    ):
        res = reviews_of_customers_with_over_1000_reviews.describe().compute()
        res.to_csv(
            f"{CSV_DIR}/describe_reviews_of_customers_with_over_1000_reviews.csv"
        )

if not os.path.exists(f"{CSV_DIR}/reviews_of_customers_with_over_5000_reviews"):
    grouped = read_data(f"{CSV_DIR}/group_size_customer_id")
    grouped1 = grouped[grouped["count"] > 5000]
    reviews_of_customers_with_over_5000_reviews = (
        df.merge(grouped1, on="customer_id", how="inner")
        .drop("count", axis=1)
        .persist()
    )
    progress(reviews_of_customers_with_over_5000_reviews)
    write_data(
        reviews_of_customers_with_over_5000_reviews,
        f"{CSV_DIR}/reviews_of_customers_with_over_5000_reviews",
    )
    if not os.path.exists(
        f"{CSV_DIR}/describe_reviews_of_customers_with_over_5000_reviews.csv"
    ):
        res = reviews_of_customers_with_over_5000_reviews.describe().compute()
        res.to_csv(
            f"{CSV_DIR}/describe_reviews_of_customers_with_over_5000_reviews.csv"
        )


if not os.path.exists(f"{CSV_DIR}/reviews_of_customers_with_less_than_2_reviews"):
    grouped = read_data(f"{CSV_DIR}/group_size_customer_id")
    grouped1 = grouped[grouped["count"] < 2]
    reviews_of_customers_with_less_than_2_reviews = (
        df.merge(grouped1, on="customer_id", how="inner")
        .drop("count", axis=1)
        .persist()
    )
    progress(reviews_of_customers_with_less_than_2_reviews)
    write_data(
        reviews_of_customers_with_less_than_2_reviews,
        f"{CSV_DIR}/reviews_of_customers_with_less_than_2_reviews",
    )
    if not os.path.exists(
        f"{CSV_DIR}/describe_reviews_of_customers_with_less_than_2_reviews.csv"
    ):
        res = reviews_of_customers_with_less_than_2_reviews.describe().compute()
        res.to_csv(
            f"{CSV_DIR}/describe_reviews_of_customers_with_less_than_2_reviews.csv"
        )


# df = read_data(f"{CSV_DIR}/reviews_of_customers_with_over_1000_reviews")
# df = df.loc[0:10000]
# df.to_csv(f"{CSV_DIR}/reviews_of_customers_with_over_1000_reviews_10000.csv", index=False)


# check for bots


if not os.path.exists(f"{CSV_DIR}/data_parquet_with_time_diffs"):
    # this is heavyload pandas code, as dask fails to perform similar operations
    dfc = read_data(DATA_DIR)
    dfc["customer_id"] = dfc["customer_id"].astype("str")
    dfc["review_date"] = dfc["review_date"].astype("datetime64[ns]")
    dfc = dfc[["customer_id", "review_date", "review_id", "product_id"]].compute()
    dfc = dfc.sort_values("review_date").reset_index(drop=True)
    dfc["time_diff_same_customer"] = dfc.groupby("customer_id")["review_date"].apply(
        lambda x: x.diff().dt.total_seconds()
    )
    dfc["time_diff_same_product"] = dfc.groupby("product_id")["review_date"].apply(
        lambda x: x.diff().dt.total_seconds()
    )
    from_pandas(dfc, npartitions=100).repartition(partition_size="100MB").to_parquet(
        f"{CSV_DIR}/data_parquet_with_time_diffs", engine="pyarrow"
    )


dfc = read_data(f"{CSV_DIR}/data_parquet_with_time_diffs")
count_bar_plot(dfc, "time_diff_same_customer")
count_bar_plot(dfc, "time_diff_same_product")


if not os.path.exists(f"{CSV_DIR}/max_reviews_per_customer.csv"):
    df = read_data(DATA_DIR)
    df2 = (
        df.groupby(["review_date", "customer_id"])
        .review_id.count()
        .rename("review_count")
        .persist()
    )
    progress(df2)
    df2 = (
        df2.groupby("customer_id").max().rename("max_review_count").to_frame().persist()
    )
    progress(df2)
    df2 = (
        df2.groupby("max_review_count")
        .size()
        .rename("customer_count")
        .to_frame()
        .compute()
        .sort_index()
        .reset_index()
    )
    df2.to_csv(f"{CSV_DIR}/max_reviews_per_customer.csv", index=False)


if not os.path.exists(f"{CSV_DIR}/number_of_bot_reviews.txt"):
    df = read_data(DATA_DIR)
    df2 = (
        df.groupby(["review_date", "customer_id"])
        .review_id.count()
        .rename("review_count")
        .persist()
    )
    df2 = df2[df2 > 30].persist()
    df2 = (
        df2.groupby("customer_id").max().rename("max_review_count").to_frame().persist()
    )

    with open(f"{CSV_DIR}/number_of_bot_reviews.txt", "w") as f:
        x = len(df[df.customer_id.isin(df2.index.compute())].compute())
        print(x)
        f.write(str(x))


if not os.path.exists(f"{CSV_DIR}/max_reviews_per_customer.png"):
    plt.figure(figsize=(10, 10))
    df2 = pd.read_csv(f"{CSV_DIR}/max_reviews_per_customer.csv")
    df2["group"] = pd.qcut(df2["max_review_count"], 20, precision=0)
    df2 = df2.groupby("group").sum()
    sns.barplot(
        data=df2, x=df2.index, y=df2["customer_count"], palette="crest", errorbar=None
    )
    plt.xlabel("Max reviews per day")
    plt.ylabel("Number of customers")
    plt.xticks(rotation=90, fontsize=8)
    plt.yscale("log")
    plt.savefig(f"{TARGET_DIR}/max_reviews_per_customer.png", bbox_inches="tight")
    plt.clf()


# alterativeley

for i in ["time_diff_same_customer", "time_diff_same_product"]:
    dfc2 = load_group_size_csv(dfc, [i])

    if not os.path.exists(f"{TARGET_DIR}/group_size_{i}2.png"):
        plt.figure(figsize=(10, 10))
        dfc2["group"] = pd.cut(
            dfc2.index,
            bins=[
                -0.001,
                0,
                86400,
                172800,
                259200.00000000003,
                345600,
                432000,
                518400.00000000006,
                max(dfc2.index.values),
            ],
            precision=0,
        )
        dfc2 = dfc2.groupby("group").sum()
        sns.barplot(
            data=dfc2,
            x=dfc2.index.values,
            y=dfc2["count"],
            palette="crest",
            errorbar=None,
        )
        plt.xlabel(i)
        plt.ylabel("Number of reviews")
        plt.xticks(rotation=90, fontsize=8)
        plt.yscale("log")
        plt.savefig(f"{TARGET_DIR}/group_size_{i}2.png", bbox_inches="tight")
        plt.clf()


# time

dfc = load_group_size_csv(df, ["review_date"])

if not os.path.exists(f"{CSV_DIR}/group_size_review_year.png"):
    dfc["review_date"] = pd.to_datetime(dfc.index)
    dfc["year"] = dfc["review_date"].dt.year
    dfc_yearly = dfc.groupby(["year"])["count"].sum().reset_index()
    plt.figure(figsize=(10, 10))
    sns.barplot(data=dfc_yearly, x="year", y="count", palette="crest", errorbar=None)
    plt.yscale("log")
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Year")
    plt.ylabel("Number of reviews")
    plt.savefig(f"{TARGET_DIR}/group_size_review_year.png", bbox_inches="tight")
    plt.clf()

# product category

dfc = load_group_size_csv(df, ["product_category"])

if not os.path.exists(f"{TARGET_DIR}/group_size_product_category.png"):
    dfc = dfc.sort_values(by="count", ascending=False)
    plt.figure(figsize=(10, 10))
    sns.barplot(x=dfc.index, y=dfc["count"], palette="crest")
    plt.xlabel("product_category")
    plt.ylabel("Number of reviews")
    plt.xticks(rotation=90, fontsize=8)
    plt.yscale("log")
    plt.savefig(f"{TARGET_DIR}/group_size_product_category.png", bbox_inches="tight")
    plt.clf()


# create describe for columns

df_pandas = df.compute()

columns = ["product_category", "star_rating", "vine", "verified_purchase"]
combinations = []
for r in range(1, len(columns) + 1):
    for combination in itertools.combinations(columns, r):
        combinations.append(list(combination))

for column in combinations:
    if not os.path.exists(f"{CSV_DIR}/describe_column_{column}.csv"):
        df_pandas.groupby(column).describe().to_csv(
            f"{CSV_DIR}/describe_column_{column}.csv"
        )
