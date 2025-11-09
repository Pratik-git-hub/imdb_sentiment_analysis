import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import streamlit as st

#load model
model = load_model('IMDB_RNN/simple_rnn_imdb.h5')

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#Step 2 : Helper Functions
# function to decode reviews
def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text) :
    words= text.lower().split()
    encoded_review = [word_index.get(word,2) +3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
## predict
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

##Step3
st.title("IMDB movie sentiment analysis")
st.write("Enter the movie review to classify it as positive of negative")
example_review = st.text_area("Enter review")

if st.button("Classify"):
    sentiment , prediction = predict_sentiment(example_review)
    st.write("Sentiment for the review is: {}".format(sentiment))
    st.write("Prediction: {}".format(prediction))