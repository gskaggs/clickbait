import pandas as pd
import numpy as np
import pickle
from pickle import load
from pickle import dump
from scipy import sparse
import nltk
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings

archive_df = pd.read_csv("./hackathon_data/upworthy-archive.csv", parse_dates=["created_at", "updated_at"], low_memory=False)

#loading pickled model
model = pickle.load(open('nbmodel.pkl','rb'))
stopwords_list = stopwords.words('english')
#loading tfidf vectorizer
vectorizer=load(open('tfidf.pkl','rb'))
#setting up functions to clean text and create engineered features with new data
#@st.cache
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    #text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('“','',text)
    text = re.sub('”','',text)
    text = re.sub('’','',text)
    text = re.sub('–','',text)
    text = re.sub('‘','',text)

    return text

def contains_question(headline):
    if "?" in headline or headline.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do')):
        return 1
    else:
        return 0

def contains_exclamation(headline):
    if "!" in headline:
        return 1
    else:
        return 0

def starts_with_num(headline):
    if headline.startswith(('1','2','3','4','5','6','7','8','9')):
        return 1
    else:
        return 0

def analyze_headline(sentence):
    cleaned_sentence = clean_text_round1(sentence)
    headline_words = len(cleaned_sentence.split())
    question = contains_question(cleaned_sentence)
    exclamation = contains_exclamation(cleaned_sentence)
    starts_num = starts_with_num(cleaned_sentence)
    input=[cleaned_sentence]
    vectorized = vectorizer.transform(input)
    final = sparse.hstack([question,exclamation,starts_num,headline_words,vectorized])
    result = model.predict(final)
    return result



winners_df = archive_df[archive_df["winner"] == True]
headlines = list(winners_df["package_headline"])

scores = [int(analyze_headline(headline)) for headline in headlines]
scores = pd.DataFrame(scores)
scores.columns = ['clickbait']
print(scores.head())

winners_df = pd.concat([winners_df.reset_index(drop=True), scores], axis=1)
print(winners_df.head())
winners_df["week"] = winners_df["test_week"].apply(lambda x: (x // 100 - 2013) * 52 + x % 100)


results = {}
for idx, row in archive_df.iterrows():
    if not row['winner']:
        continue
    headline = row['package_headline']
    week = row['test_week']
    results[week] = results.get(week, (0, 0))
    click_bait = int(analyze_headline(headline))
    click_bait_count, N = results[week]
    click_bait_count += click_bait
    N += 1
    results[week] = (click_bait_count, N)

