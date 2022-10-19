from itertools import count
import textstat
import pandas as pd
import matplotlib.pyplot as plt
from analysis_core import apply, load_reviews_from_folder, filter_dataframe

def saveplot(array, name, bins=20):
    fig, ax = plt.subplots()
    ax.set_title(f"Median: {array.median()} | Mean: {array.mean()}")
    ax.hist(array, bins=bins, linewidth=0.5, edgecolor="white")
    ax.set_yscale('log', base=10)
    plt.savefig(name)


def contains_https_http_link(row):
    count =  row.review_body.lower().count('https://') +  row.review_body.lower().count('http://')
    if count > 0:
        return True
    else:
        return False

reviews = load_reviews_from_folder('./data', nrows=100000)
#sentence_count = apply(reviews, lambda review: textstat.sentence_count(review), 'review_body', exception_value=0)
#poly_syllabcount = apply(reviews, lambda review: textstat.polysyllabcount(review), 'review_body', exception_value=0)
text_complexity = apply(reviews, lambda review: textstat.flesch_reading_ease(review), 'review_body', skip_on_exception=True)

reviews_with_links = filter_dataframe(reviews, contains_https_http_link, skip_on_exception=True)
text_complexity_with_links = apply(reviews_with_links, lambda review: textstat.flesch_reading_ease(review), 'review_body', skip_on_exception=True)

text_complexity = filter_dataframe(text_complexity, lambda x: x > -50)
text_complexity_with_links = filter_dataframe(text_complexity_with_links, lambda x: x > -50)
saveplot(text_complexity_with_links, "text_complexity_review_with_links", bins=100)
saveplot(text_complexity, "text_complexity", bins=100)

#saveplot(youtube_links, "youtube_links", bins=10)

#saveplot(sentence_count, "sentences")
#saveplot(text_complexity, "complexity" )
#saveplot(poly_syllabcount, "syllabcount")
#saveplot(youtube_links, "youtube_links", bins=10)
