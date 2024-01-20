import streamlit as st
import pandas as pd
import numpy as np

import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# 1. Preprocessing
if st.button('Predict'):

    # 2. Transform Text
    transformed_sms = transform_text(input_sms)
    # 3. Vectorize Text
    vector_input = tfidf.transform([transformed_sms])
    # 4. Predict
    result = model.predict(vector_input)[0]
    # 5. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

st.header("METRICS")
st.write("Accuracy: 97%")
st.write("Precision: 1.0")
