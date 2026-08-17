import nltk
from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

porter = PorterStemmer()


# stem: Walk
porter.stem("walking")
porter.stem("walked")
porter.stem('Walks')

# stem: run
porter.stem('ran') # issue
porter.stem('running')

# stem: buss
porter.stem('busses')

# stem: replacement
porter.stem('replacement')


# lemmatization
sentence = "Lemmatization is more sophisticated than stemming".split()

for token in sentence:
    print(porter.stem(token), end=' ')


# stem e.g.
porter.stem("unnecessary")
porter.stem('berry')



nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize("walking")
lemmatizer.lemmatize("walking", pos = wordnet.VERB)

lemmatizer.lemmatize('going')
lemmatizer.lemmatize('going', pos = wordnet.VERB)

lemmatizer.lemmatize('ran', pos = wordnet.VERB)


porter.stem('mice')

porter.stem('was')
lemmatizer.lemmatize('was', wordnet.VERB)


porter.stem('is')
lemmatizer.lemmatize('is', wordnet.VERB)


porter.stem('better')
lemmatizer.lemmatize('better', wordnet.ADJ)

# part of speech tagging


