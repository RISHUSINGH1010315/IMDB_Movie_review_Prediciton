import streamlit as st
from transformers import pipeline

# Load sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review to classify it as **Positive or Negative**.")

# User input
user_input = st.text_area("Movie Review")

# Button
if st.button("Classify Review"):

    if user_input.strip() != "":

        result = sentiment_pipeline(user_input)[0]

        sentiment = result["label"]
        score = result["score"]

        if sentiment == "POSITIVE":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")

        st.write(f"Confidence Score: {score:.4f}")

    else:
        st.warning("Please enter a movie review.")
