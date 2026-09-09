## text preprocessing

import pandas as pd
import spacy
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer

### create dataset
data = [
    "When life gives you lemons, make lemonade!",
    "She bought 2 lemons for $1 at Maven Market.",
    "A dozen lemons will make a gallon of lemonade. [AllRecipes]",
    "Lemon, lemon, lemons, lemon, lemon, lemons",
    "He's running to the market to get a lemon - there's a great sale today.",
    "Does Maven Market carry Eureka lemons or Meyer lemons?",
    "An Arnold Palmer is half lemonade, half iced tea. [Wikipedia]",
    "iced tea is my favorite"
]

pd.set_option('display.max_colwidth', None)
df_text = pd.DataFrame(data, columns=['sentences'])


test = [
"We're going to start this course with traditional NLP applications.",
"Then we'll move on to modern NLP theory.",
"Finally, we'll wrap things up with modern NLP applications."
]

# why Series not Dataframe
# because it is just one single format but dataframes are more flexible
test = pd.Series(test)


## text preprocessing with pandas

df = df_text.copy()

## making text lowercase
df['sentences_clean'] = df['sentences'].str.lower()

## cleaning everything between brackets

df["sentences_clean"] = df["sentences_clean"].str.replace(r'\[.*?\]','',regex=True)

## remove punctuations

df["sentences_clean"] = df["sentences_clean"].str.replace(r'[^\w\s]','',regex=True)



## function to preprocess text data for us

def preprocess_data(column):
    column = column.str.lower()
    column = column.str.replace(r'[^\w\s]','',regex=True)
    column = column.str.replace(r'\[.*?\]','',regex=True)
    return column


## Spacy
# to use spacy you need to make a spacy object
# then pass your text file to the object to be able 
# to use spacy things work for you.

## tokenization
# break text into smaller pieces 



nlp = spacy.load('en_core_web_sm')
phrase = df.sentences_clean[0]

sp_text = nlp(phrase)


## tokenize
tokenize = [token.text for token in sp_text]


## lemmatize
lemm = [token.lemma_ for token in sp_text]

## stop words
norm = [token.lemma_ for token in sp_text if not token.is_stop]

" ".join(norm)

def token_lemma_nonstop(text):
    
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    output = ' '.join(output)
    return output
        

test.apply(token_lemma_nonstop)

df.sentences
phrase2 = preprocess_data(df.sentences).apply(token_lemma_nonstop)[0]
doc2 = nlp(phrase2)

[(token.text, token.pos_ )for token in doc2]

NOUNS = [token.text for token in doc2 if token.pos_ in ['NOUN','PROPN']]
' '.join(NOUNS)


def filter_pos(text, pos=['NOUN','PROPN']):
    doc = nlp(text)

    [(token.text, token.pos_ )for token in doc]

    output = [token.text for token in doc2 if token.pos_ in pos]
    output = ' '.join(output)
    return output

text = preprocess_data(df.sentences).apply(token_lemma_nonstop).apply(filter_pos)


#################################### PIPELINE ################################


def preprocess_data(column):
    column = column.str.lower()
    column = column.str.replace(r'[^\w\s]','',regex=True)
    column = column.str.replace(r'\[.*?\]','',regex=True)
    return column


def token_lemma_nonstop(text):
    
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    output = ' '.join(output)
    return output


def filter_pos(text, pos=['NOUN','PROPN']):
    doc = nlp(text)

    [(token.text, token.pos_ )for token in doc]

    output = [token.text for token in doc if token.pos_ in pos]
    output = ' '.join(output)
    return output  



def nlp_pipeline(series):
    output = preprocess_data(series).apply(token_lemma_nonstop).apply(filter_pos)
    return output


df.sentences
test

nlp_pipeline(test)
text_clean = nlp_pipeline(df.sentences)

pd.to_pickle(text_clean, 'text_clean.pkl')
# because it is really fast and it work really well with python
# and it is not really readible for humans




### 4. Count vectorizer
text_clean =  pd.read_pickle('text_clean.pkl')
c_v = CountVectorizer(stop_words='english', min_df=0.2, max_df=0.8, ngram_range=(1,2) )
dtm = c_v.fit_transform(text_clean)
# sparse matrix have alot of zeros in it
dtm_df = pd.DataFrame(dtm.toarray(), columns=c_v.get_feature_names_out())


## return the most common word in the corpus
### lemon
term_freq = dtm_df.sum()
term_freq.sort_values().plot(kind='barh');



## tf-idf

### term frequency: mean that what is the percentage of a word being
## in document relative to other words
## the output of tf is a percentage
## (how many times a word appeared in a document) / (number of all terms in the documents)

### Inverse document frequency: the goal of this term is to distinguish 
## important and non-important words in a document
## log ((total docs + 1)/(documents with the word + 1))

# after that we multiple the two terms together


from sklearn.feature_extraction.text import TfidfVectorizer


vect2 = TfidfVectorizer(stop_words='english', min_df=0.2, max_df=0.8, ngram_range=(1,2))
tf2 = vect2.fit_transform(text_clean)
df_tf = pd.DataFrame(tf2.toarray(), columns=vect2.get_feature_names_out())



vect = TfidfVectorizer()
tf = vect.fit_transform(text_clean)
df_tf = pd.DataFrame(tf.toarray(), columns=vect.get_feature_names_out())







#################################### Assignment pandas ################################
### text preprocessing using pandas
df_text_book = pd.read_csv('childrens_books.csv')
df = df_text_book.copy()


df['Description_clean'] = df["Description"].str.lower()
df['Description_clean'] = df['Description_clean'].str.replace('\xa0', ' ')
df['Description_clean'] = df['Description_clean'].str.replace(r'[^\w\s]', '', regex=True)




#################################### Assignment spaCy ################################


desc = df['Description_clean'][1]

nlp = spacy.load('en_core_web_sm')

phrase_child = nlp(desc)

tokenize = [token.text for token in phrase_child]
lemm = [token.lemma_ for token in phrase_child]
stop_wods = [token.lemma_ for token in phrase_child if not token.is_stop]

def nlp_child_pipe(text):
    
    phrase_child = nlp(text)
    # output = [token.text for token in phrase_child]
    # output = [token.lemma_ for token in phrase_child]
    output = [token.lemma_ for token in phrase_child if not token.is_stop] 
    output = ' '.join(output)
    return output
    
df['Description_clean'] = df.Description_clean.apply(nlp_child_pipe) 


df_child_desc = {}
for i, sentence in enumerate(df['Description_clean']):
    sent =  nlp_child_pipe(sentence)
    df_child_desc[i] = sent




#################################### Assignment Count vectorizer ################################
countv = CountVectorizer(stop_words='english', min_df=0.1)
dtm = countv.fit_transform(df['Description_clean']).toarray()
dtm_df = pd.DataFrame(dtm, columns=countv.get_feature_names_out())



dtm_df.sum().sort_values().head(10)
dtm_df.sum().sort_values().tail(10)

dtm_df.sum().sort_values().tail(10).plot(kind='barh')



#################################### Assignment tf-idf vectorizer ################################

tfv = TfidfVectorizer(stop_words='english', min_df=0.1, max_df=0.5)
tf_d = tfv.fit_transform(df['Description_clean']).toarray()
tf_df = pd.DataFrame(tf_d, columns=tfv.get_feature_names_out())

tf_df.sum().sort_values().head(10)
tf_df.sum().sort_values().tail(10).plot(kind='barh')




### Key take aways:
# the middle steps is called NLP pipeline
