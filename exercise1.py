import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



df = pd.read_csv('movie_metadata.csv')

for col in df.columns:
    print(df[col].value_counts(normalize=True))
    print('-'*50)
    
    
df.columns


df['color']
len(df)


docs = []

for i in range(len(df)):
    doc = f'A movie by {df["director_name"].iloc[i]}  with duration of {df["duration"].iloc[i]} played by {df["actor_1_name"].iloc[i]} and {df["actor_2_name"].iloc[i]} with budget of {df["budget"].iloc[i]} in genres of {df["genres"].iloc[i]} in language of {df["language"].iloc[i]} made in {df["country"].iloc[i]} and score of {df["imdb_score"].iloc[i]}'
    docs.append(doc) 



vectorizer = TfidfVectorizer()
docs_t = vectorizer.fit_transform(docs)
df.columns


df['movie_title'] = df['movie_title'].str.strip()

movie_idx = pd.Series(df.index, index=df['movie_title'])
idx = movie_idx['Avatar']

query = docs_t[idx]
query.toarray()
scores = cosine_similarity(query, docs_t)
scores = scores.flatten()
plt.plot(scores);


(-scores).argsort()

plt.plot(scores[(-scores).argsort()]);


# we did not start from zero becuase the first one is the movie itself
recommended_idx = (-scores).argsort()[1:6] 
df['movie_title'].iloc[recommended_idx]


def recommend(movie_title):
    
    
    vectorizer = TfidfVectorizer()
    docs_t = vectorizer.fit_transform(docs)
    
    
    df['movie_title'] = df['movie_title'].str.strip()

    movie_idx = pd.Series(df.index, index=df['movie_title'])
    
    idx = movie_idx[movie_title]
    
    query = docs_t[idx]
    scores = cosine_similarity(query, docs_t)
    
    scores = scores.flatten()
    
    recommended_idx = (-scores).argsort()[1:6]
    
    return df['movie_title'].iloc[recommended_idx]


recommend('Interstellar')

