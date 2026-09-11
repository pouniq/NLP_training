import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
pd.set_option('display.max_colwidth', None)
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
top_ten_light = df_full.sort_values('compound').tail(10)
top_ten_dark = df_full.sort_values('compound').head(10) 
