# %%%
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help="Sample file without .tsv extension")
parser.add_argument('--number_of_reviews', '-n', help="The number of reviews in the output manifest. Shuffled randomly")

args = parser.parse_args()
tsv_file = args.file
df = pd.read_csv(f'{tsv_file}.tsv', sep='\t')

json_df = json.loads(df.to_json(orient='records'))
json_df

# %%
import random

total_output_reviews = len(json_df)

if args.number_of_reviews is not None:
    number_of_reviews = int(args.number_of_reviews)
    if number_of_reviews < total_output_reviews:
        total_output_reviews = number_of_reviews
    random.shuffle(json_df)
# %%%

manifest = []
SOURCE_COLUMN = 'review_body'
METADATA_COLUMNS = ['review_id', 'product_id', 'product_title', 'review_date', 'product_category']
for i, review in enumerate(json_df[:total_output_reviews]):
    metadata = review.copy()
    del metadata['review_body']
    datapoint = {
        "source" : review[SOURCE_COLUMN],
        "metadata" : {
            'total_reviews' : total_output_reviews,
            'current_review_index' : i
        }
    }
    for column in METADATA_COLUMNS:
        datapoint['metadata'][column] = review[column]

    manifest.append(datapoint)

manifest
# %%
import time
json_string_dataset = [json.dumps(row, ensure_ascii=False) for row in manifest]
formatted_json_string = '\n'.join(json_string_dataset)

unix_timestamp = str(int(time.time()))
with open(f'{tsv_file}-{unix_timestamp}.manifest.jsonl', 'w', encoding='utf8') as file:
    file.write(formatted_json_string)