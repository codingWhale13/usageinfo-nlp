import textstat
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

sentence_count = np.array([])
word_count = np.array([]) 
text_complexity = np.array([])
poly_syllabcount = np.array([]) 

def saveplot(array, name):
    fig, ax = plt.subplots()
    ax.hist(array, bins=100, linewidth=0.5, edgecolor="white")
    ax.set_yscale('log', base=10)
    plt.savefig(name)



def analyse_data(df):
    global sentence_count
    global text_complexity
    global poly_syllabcount
    for row in df.itertuples():
        text = row.review_body
        try:
            sentencecount = textstat.sentence_count(text)
            textcomplexity = textstat.flesch_reading_ease(text)
            polysyllabcount = textstat.polysyllabcount(text)
        except:
            sentencecount = 0
            textcomplexity = 0
            polysyllabcount = 0
        sentence_count = np.append(sentence_count, sentencecount)
        text_complexity = np.append(text_complexity, textcomplexity)
        poly_syllabcount = np.append(poly_syllabcount, polysyllabcount)

  
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


saveplot(sentence_count, "sentences")
saveplot(text_complexity, "complexity" )
saveplot(poly_syllabcount, "syllabcount")