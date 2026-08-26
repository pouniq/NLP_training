import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


df_books = pd.read_csv('/Users/pouniq/nlp_train/books_data/BooksDatasetClean.csv')
df_books


docs = []
for i in range(len(df_books)):
    doc = f'book with name of {df_books['Title'].iloc[i]} {df_books['Authors'].iloc[i]} in category of {df_books['Category'].iloc[i]} was published in {df_books['Publish Date (Year)'].iloc[i]} with price of {df_books['Price Starting With ($)'].iloc[i]}'
    docs.append(doc)

vectorize = TfidfVectorizer()
docs_t = vectorize.fit_transform(docs)

df_books['Title'] = df_books['Title'].str.strip()

books_idx = pd.Series(df_books.index, index=df_books['Title'])
idx = books_idx['Tongues: To Speak or Not to Speak']
query = docs_t[idx]
query.toarray()
scores = cosine_similarity(query, docs_t)
scores = scores.flatten()
plt.plot(scores)

plt.plot(scores[(-scores).argsort()])

recom = (-scores).argsort()[1:6]
df_books['Title'].iloc[recom]
