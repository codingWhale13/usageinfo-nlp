# %%
import numpy as np


t = np.array(texts)
len(np.unique(t))

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(sentences):
    return sentences


texts = [
    [1, 24, 28],
    [1, 24, 27, 28, 28],
]

count_vect = TfidfVectorizer(tokenizer=tokenize, lowercase=False)

X_train_counts = count_vect.fit_transform(texts)

cosine_similarity(X_train_counts.toarray())
# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

mlb.fit_transform(texts)
