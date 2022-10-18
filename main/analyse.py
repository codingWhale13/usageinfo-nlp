import textstat
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

sentence_count = np.array([])


def analyse_data(df):
    global sentence_count
    for row in df.itertuples():
        text = row.review_body
        try:
            count = textstat.sentence_count(text)
            if count > 100:
                print(text)
        except:
            count = 0
        sentence_count = np.append(sentence_count, count)
  
r_folder = "./data"
w_folder = "./data_cleaned"

for filename in os.listdir(r_folder):
    if filename.endswith(".tsv"):
        tsv_input = pd.read_csv(r_folder + "/" + filename, sep='\t', nrows=5000)
        print(filename)
        analyse_data(tsv_input)

        continue
    else:
        continue

print(sentence_count[:20])
fig, ax = plt.subplots()

ax.hist(sentence_count, bins=100, linewidth=0.5, edgecolor="white")
ax.set_yscale('log', base=10)
plt.savefig('sentence2')