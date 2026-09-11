import pandas as pd
import spacy


nlp = spacy.load('en_core_web_sm')
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

if __name__ == "main":
    print('text preprocessing module can be used.')