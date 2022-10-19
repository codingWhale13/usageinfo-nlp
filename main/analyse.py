import textstat
import pandas as pd
import matplotlib.pyplot as plt
from analysis_core import apply, load_reviews_from_folder

def saveplot(array, name):
    fig, ax = plt.subplots()
    ax.hist(array, bins=100, linewidth=0.5, edgecolor="white")
    ax.set_yscale('log', base=10)
    plt.savefig(name)

reviews = load_reviews_from_folder('./data', nrows=5000)
sentence_count = apply(reviews, lambda review: textstat.sentence_count(review), 'review_body', exception_value=0)
poly_syllabcount = apply(reviews, lambda review: textstat.polysyllabcount(review), 'review_body', exception_value=0)
text_complexity = apply(reviews, lambda review: textstat.flesch_reading_ease(review), 'review_body', skip_on_exception=True)

saveplot(sentence_count, "sentences")
saveplot(text_complexity, "complexity" )
saveplot(poly_syllabcount, "syllabcount")