import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
import streamlit as st


# -----------------------------
# FORCE FIX FOR KERAS 3
# -----------------------------
class CustomSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)   # remove unsupported argument
        super().__init__(*args, **kwargs)


# -----------------------------
# LOAD WORD INDEX
# -----------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# -----------------------------
# LOAD MODEL (FORCE)
# -----------------------------
model = load_model(
    "simple_rnn_imdb.h5",
    compile=False,
    custom_objects={"SimpleRNN": CustomSimpleRNN}
)


# -----------------------------
# DECODE REVIEW
# -----------------------------
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# -----------------------------
# PREPROCESS TEXT
# -----------------------------
def preprocess_text(text):
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.word_index = word_index
    
    sequence_text = tokenizer.texts_to_sequences([text])
    padded = sequence.pad_sequences(sequence_text, maxlen=500)
    
    return padded


# -----------------------------
# PREDICT SENTIMENT
# -----------------------------
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review to classify it as Positive or Negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip() != "":
        sentiment, score = predict_sentiment(user_input)

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score}")
    else:
        st.write("Please enter a movie review.")