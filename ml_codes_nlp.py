import pandas as pd
import openpyxl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


## Naive bayes import
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score




from preprocess_text_function import nlp_pipeline
# create a list of sentences
data = [
    "When life gives you lemons, make lemonade! 🙂",
    "She bought 2 lemons for $1 at Maven Market.",
    "A dozen lemons will make a gallon of lemonade. [AllRecipes]",
    "Lemon, lemon, lemons, lemon, lemon, lemons",
    "He's running to the market to get a lemon - there's a great sale today.",
    "iced tea is my favorite",
    "I didn't like the taste of that lemonade at all.",
    "My lemons went bad before I could use them, unfortunately."
]
# expand the column width to see the full sentences
# pd.set_option('display.max_colwidth', None)
# turn it into a dataframe
data_df = pd.DataFrame(data, columns=["sentence"])
data_df.head()
# make a copy of the dataframe
df = data_df.copy()

df.head()

test = df.sentence[0]
model = SentimentIntensityAnalyzer()
model.polarity_scores(test)

model.polarity_scores(test)['compound']

def get_sentiment(text):
   model = SentimentIntensityAnalyzer()
   output = model.polarity_scores(text)['compound']
   return output
    

df['sentiment_compound'] = df.sentence.apply(get_sentiment)


############################assignment movie reviews############################


data_movie = pd.read_csv('./data/movie_reviews.csv')
df = data_movie.copy(deep=True)

text_movie_info = df['movie_info']
sen_model = SentimentIntensityAnalyzer()
sen_model.polarity_scores(text_movie_info[0])


sen_model = SentimentIntensityAnalyzer()
def sentiment_movie_info(text):
    output = sen_model.polarity_scores(text)
    return output

sentiment_movie = df["movie_info"].apply(sentiment_movie_info)



compound = {}
for i in range(len(sentiment_movie)):
    
    output = sentiment_movie.iloc[i]['compound']
    compound[i] = output
    
    
comp_df = pd.DataFrame(
    {
        'compound': compound
    }
)

df_full = pd.concat([df, comp_df], axis=1)
top_ten_light = df_full[['movie_title', 'compound']].sort_values('compound').tail(10)
top_ten_dark = df_full[['movie_title', 'compound']].sort_values('compound').head(10)



########################## countvectorizer + naive bayes ##########################################

data_review = pd.read_excel('./data/Popchip_Reviews.xlsx')
data_review.head(2)
data_review.shape

data_review.groupby('Priority').count()['Id']
data_review.Priority.value_counts()

data_review['Text_clean'] = nlp_pipeline(data_review['Text'])


## Count vectorizer
### min_df is saying that return values (words) that are 
### atleast in 20 percent of all the documents (corpus)
ctv = CountVectorizer(stop_words='english',min_df=0.2, ngram_range=(1,2))
X = ctv.fit_transform(data_review['Text_clean'])
X_df = pd.DataFrame(X.toarray(), columns=ctv.get_feature_names_out())
y = data_review['Priority']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))



new_reviews = pd.Series([
    "Pop chips are my favorite! I love these chips so much.",
    "Taste bad. I don't like the flavor options or taste.",
    "Solid snack."])

test_review = nlp_pipeline(new_reviews)
test_review_clean = pd.DataFrame(ctv.transform(test_review).toarray(),columns=ctv.get_feature_names_out())
y_test_review_clean = model.predict(test_review_clean)


########################## tfidf + logisticRegression ##########################################

tv = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1)
Xt = tv.fit_transform(data_review['Text_clean'])
Xt_df = pd.DataFrame(Xt.toarray(), columns=tv.get_feature_names_out())
y


X_train, X_test, y_train, y_test = train_test_split(Xt,y, test_size=0.2, random_state=42)
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)

y_pred = model_lr.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))



data_review.head()
data_review['pred_nb'] = model.predict_proba(X_df)[:,0]
data_review['pred_lr'] = model_lr.predict_proba(Xt_df)[:,0]

data_review.sort_values(by='pred_nb', ascending=False)


## next step is to fine tune your model
## - text preprocessing
## - vectorization
## - feature engineering
## - modelling -- trying different probabaility cut-off points

df_movie_reviews = pd.read_csv('./data/movie_reviews.csv')

df_movie_reviews.groupby('director_gender').count()

X = df_movie_reviews['movie_info']
y = df_movie_reviews['director_gender']

X_clean = nlp_pipeline(X)

cv = CountVectorizer(stop_words='english', min_df=0.1, ngram_range=(1,2))
X_clean = cv.fit_transform(X_clean)
X_clean = pd.DataFrame(X_clean.toarray(), columns = cv.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(X_clean,y, test_size=0.2, random_state=42 ,shuffle=True)


model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred = model_nb.predict(X_train)


y_pred_test = model_nb.predict(X_test)

print(classification_report(y_train, y_pred))
print(classification_report(y_test, y_pred_test))


