import pandas as pd
import os



def filter_data(df):
    df.dropna()
    print(df.shape)
    df
    df.drop(index=df[df['review_body'] == 'Good'].index, inplace=True)
    #df = df[df.review_body != ':)']
    df.drop_duplicates(subset=["review_id"], keep="first", inplace=True)
    print(df.shape)
    
r_folder = "./data"
w_folder = "../data_cleaned"

for filename in os.listdir(r_folder):
    if filename.endswith(".tsv"):
        tsv_input = pd.read_csv(r_folder + "/" + filename, sep='\t', nrows=100000)
        print(filename)
        filter_data(tsv_input)

        continue
    else:
        continue