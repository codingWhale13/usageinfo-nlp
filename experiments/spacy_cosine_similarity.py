import numpy as np
import spacy
from scipy import spatial

"""
This experiment makes sure that using the `similarity` method on Spacy `Doc` objects
yields the same result (up to rounding errors) as  computing the cosine similarity "by hand".
"""

nlp = spacy.load("en_core_web_md")

# Get the vector for a text using spacy
text1 = "working in the yard"
text2 = "carrying umbrellas to beach"
doc1 = nlp(text1)
doc2 = nlp(text2)

# Get the vector for each word in the document
vecs1 = [word.vector for word in doc1]
vecs2 = [word.vector for word in doc2]

# Average the vectors for the entire document
doc1_vec = np.mean(vecs1, axis=0)
doc2_vec = np.mean(vecs2, axis=0)

# Compute the cosine similarity between the two vectors
similarity_1 = doc1.similarity(doc2)
similarity_2 = 1 - spatial.distance.cosine(doc1_vec, doc2_vec)

print(f"With `similarity` method: {similarity_1}")
print(f"With manual cosine similarity; {similarity_2}")

assert abs(similarity_1 - similarity_2) < 0.00001
