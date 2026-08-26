import pandas as pd
import numpy as np
import nltk

from nltk import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
df = pd.read_csv('bbc_text_cls.csv')
df


idx = 0
word2int = {}
docs = []

for doc in df['text']:
    words = word_tokenize(doc.lower())
    doc_to_int = []
    for word in words:
        if word not in word2int:
            word2int[word] = idx
            idx += 1
        
        doc_to_int.append(word2int[word])
    docs.append(doc_to_int)
    
idx2word = {v:k for k,v in word2int.items()}

# number of docs
N = len(df['text'])
V = len(word2int)


tf = np.zeros((N, V))


for i, doc_as_int in enumerate(docs):
    for j in doc_as_int:
        tf[i,j] += 1
        

doc_freq = np.sum(tf > 0 , axis = 0)
idf = np.log(N / doc_freq) 

tf.shape
idf.shape

tf_idf = tf * idf

tf_idf.shape

np.random.seed(42)

i = np.random.choice(N)
row = df.iloc[i]
print(row["labels"])
print('-'*20)

row['text'].split('\n',1)[0]
scores = tf_idf[i]
print(row['text'])
print('-'*20)
indicies = (-scores).argsort()
for j in indicies[:5]:
    print(idx2word[j])
    
