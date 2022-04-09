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

weeks = archive_df['test_week']
min_weeks = min(weeks)
num_weeks = max(weeks) - min_weeks + 1
percent_click_bait = [0] * num_weeks
click_bait, num_headslines = [0] * num_weeks, [0] * num_weeks
for k in results:
    week_num = k - min_weeks
    click_bait_count, N = results[k]
    click_bait[week_num], num_headslines[week_num] = click_bait_count, N
    percent_click_bait[week_num] = click_bait_count * 100 / N

print(percent_click_bait[:5])
y, x = percent_click_bait, list(range(num_weeks))
with open('percent_click_bait.pkl', 'wb') as handle:
    pickle.dump((percent_click_bait, click_bait, num_headslines), handle)
plt.plot(x, y)
# plt.show()