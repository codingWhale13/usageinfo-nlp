from pandas import DataFrame, get_dummies, concat


def one_hot_encode(df: DataFrame, *columns: list[str]) -> DataFrame:
    new_df = df.copy().drop(columns=list(columns))
    for column in columns:
        new_df = concat([new_df, get_dummies(df[column])], axis=1)
    return new_df
