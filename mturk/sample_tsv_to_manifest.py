# %%%
import pandas as pd
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help="Sample file without .tsv extension")
parser.add_argument('--number_of_reviews', '-n', help="The number of reviews in the output manifest. Shuffled randomly")
parser.add_argument('--reviews_per_task', '-b', default= 5, type=int, help="The number of reviews for each task", )


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
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

manifest = []
SOURCE_COLUMN = 'review_body'
METADATA_COLUMNS = ['review_id', 'product_id', 'product_title', 'review_date', 'product_category']

reviews_per_task = args.reviews_per_task

for reviews_batch in grouper(json_df[:total_output_reviews], reviews_per_task):
    source = []
    metadata = []
    for review in reviews_batch:
        review_metadata = review.copy()
        del review_metadata['review_body']
        source.append(review[SOURCE_COLUMN])
        
        datapoint_metadata = {}
        for column in METADATA_COLUMNS:
            datapoint_metadata[column] = review[column]
        metadata.append(datapoint_metadata)

    manifest.append({
    "source" : source,
    "metadata" : metadata
   })

manifest

# %%
from create_hit import create_hit_question_xml
with open('question.xml','w') as file:
    source = json.dumps(manifest[0]["source"])
    metadata = json.dumps(manifest[0]['metadata'])
    xml = create_hit_question_xml(source, metadata)
    file.write(str(xml))
# %%
import time
json_string_dataset = [json.dumps(row, ensure_ascii=False) for row in manifest]
formatted_json_string = '\n'.join(json_string_dataset)

unix_timestamp = str(int(time.time()))
with open(f'{tsv_file}-{unix_timestamp}.manifest.jsonl', 'w', encoding='utf8') as file:
    file.write(formatted_json_string)