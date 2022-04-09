import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pickle import load
from scipy import sparse
import nltk
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
warnings.filterwarnings('ignore')




#st.markdown(‘Clickbait Detector’)
#creating title of app
# st.title('The  Detector')

st.markdown("# The Upworthy Clickbait Detector ")
st.markdown("Our dashboard lets you test wether an article from upworthy.com"
            " is click bait or not. Just"
             " submit the headline that you wish to test in the following text field:")

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

#creating area to submit headline
sentence = st.text_area('Enter headline')
#once submitted, headline is cleaned and features are engineered, tfidf is applied
#model takes in sparse matrix of tfidf values and engineered features

if st.button('Submit'):
    cleaned_sentence = clean_text_round1(sentence)
    headline_words = len(cleaned_sentence.split())
    question = contains_question(cleaned_sentence)
    exclamation = contains_exclamation(cleaned_sentence)
    starts_with_num = starts_with_num(cleaned_sentence)
    input=[cleaned_sentence]
    vectorized = vectorizer.transform(input)
    final = sparse.hstack([question,exclamation,starts_with_num,headline_words,vectorized])
    result = model.predict(final)

    if result == 1:
        st.error('Clickbait detected')
    else:
        st.success('Not clickbait! :-)')


st.markdown('''
This model used news articles from reputable sources The New York Times and the Washington Post, as well as clickbait sources like Buzzfeed and Thatscoop.

We considered using the following models:
* Random Forest
* Naive Bayes
* Logistic Regression 
* Linear SVM

But ultimately we decided to use Naive Bayes, not only because it outperformed the other models in terms of recall and precision but also because of its relative speed on large datasets.

The credit for training these models goes to Alison Salerno [https://towardsdatascience.com/is-this-headline-clickbait-86d27dc9b389](https://towardsdatascience.com/is-this-headline-clickbait-86d27dc9b389)

''')

