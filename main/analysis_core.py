import pandas as pd
import os

"""
Apply a function rowwise to a pandas dataframe

label: get a label/datapoint from a row directly not the whole row
exception_value: if the apply function throws an exception return a default value
skip_on_exception: Skip rows, where the apply function throws an exception
"""
def apply(df, apply_function, label=None, exception_value=None, skip_on_exception=False):
    def apply_function_with_exception_value(row):
        try:
            return apply_function(row)
        except Exception as e:
            return exception_value

    def apply_function_with_exception_skip(row):
        try:
            return apply_function(row)
        except Exception as e:
            return None

    if skip_on_exception:
        a0 = apply_function_with_exception_skip
    elif exception_value is not None:
        a0 = apply_function_with_exception_value
    else:
        a0 = apply_function
    
    if label is not None:
        a = lambda row: a0(row[label]) 
    else:
        a = a0
    
    if type(df) == pd.Series:
        results = df.apply(a)
    else:
        results = df.apply(a, axis=1)

    if skip_on_exception:
        return results[results.notna()]
    else:
        return results


def filter_dataframe(df, apply_function, label=None, skip_on_exception = False):
    if skip_on_exception:
        return df[apply(df, apply_function, label=label, exception_value=False)]
    else:
        return df[apply(df, apply_function, label=label)]

def load_reviews_from_folder(path, nrows=None):
    dfs = []
    for filename in os.listdir(path):
        if filename.endswith(".tsv"):
            if nrows is None:
                df = pd.read_csv(path + "/" + filename, sep='\t')
            else:
                df = pd.read_csv(path + "/" + filename, sep='\t', nrows=nrows)
            dfs.append(df)
        else:
            continue

    return pd.concat(dfs)
