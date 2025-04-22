import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are downloaded
"""nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
"""
# Load the trained model
try:
    model = joblib.load("sentiment_model_svm.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load vectorizer
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check file paths.")

# Initialize the lemmatizer once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Streamlit UI
st.title("🎭 Sentiment Analysis App")

user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if model and vectorizer:
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])  # Convert input to TF-IDF
        prediction = model.predict(vectorized_text)  # Predict sentiment

        # Show result
        if prediction[0] == "positive":
            st.success("This review is **Positive**! 😊")
        else:
            st.error("This review is **Negative**. 😡")
    else:
        st.error("Model or vectorizer is not loaded properly. Please check the files.")
