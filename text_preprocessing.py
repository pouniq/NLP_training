## text preprocessing

import pandas as pd
import spacy

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




## assignment
### text preprocessing using pandas
df_text_book = pd.read_csv('childrens_books.csv')
df = df_text_book.copy()


df['Description_clean'] = df["Description"].str.lower()
df['Description_clean'] = df['Description'].str.replace('\xa0', ' ')
df['Description_clean'].iloc[0]
df['Description_clean'] = df['Description'].str.replace(r'[^\w\s]', '', regex=True)



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
