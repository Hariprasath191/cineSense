# 🎭 CineSense – Movie Review Sentiment Analysis

**CineSense** is an interactive web application that predicts the sentiment of movie reviews. Users can enter a review, and the app classifies it as **Positive** or **Negative** using a trained machine learning model.

The app is built using **Python**, **Streamlit**, and **scikit-learn**, and uses natural language processing (NLP) techniques to analyze text data.

---

## 🛠 Features

* **Text Input**: Enter a movie review.
* **Sentiment Prediction**: Classifies reviews as **Positive** or **Negative**.
* **NLP Preprocessing**: Tokenization, stopword removal, and lemmatization.
* **TF-IDF Vectorization**: Converts text into numerical features for the model.
* **Interactive UI**: Streamlit-based web interface for easy interaction.

---

## 📂 Project Structure

```
cineSense/
│
├── app.py                      # Main Streamlit application
├── sentiment_model_svm.pkl     # Pre-trained SVM model
├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/Hariprasath191/cineSense.git
cd cineSense
```

2. **Create and activate a virtual environment** (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

5. **Open the web app** in your browser (usually `http://localhost:8501`).

---

## 🧪 How It Works

1. User enters a movie review.
2. The app preprocesses the text:

   * Converts to lowercase
   * Removes special characters
   * Tokenizes words
   * Removes stopwords
   * Lemmatizes words
3. The processed text is converted to a TF-IDF vector.
4. The trained SVM model predicts the sentiment.
5. The app displays the result as **Positive** or **Negative**.

---

## 🔧 Technology Stack

* **Frontend / Web App:** Streamlit
* **Backend / ML:** Python, scikit-learn
* **NLP Processing:** NLTK
* **Model:** SVM classifier trained on movie review data

---

## 📈 Future Enhancements

* Support **neutral sentiment** classification.
* Expand to **multiple languages**.
* Add **bulk review analysis** via file upload.
* Deploy online using **Streamlit Cloud** or **Heroku**.
* Improve accuracy using **deep learning models** like LSTM or BERT.

---

## ⚡ Usage

* Launch the app using Streamlit.
* Enter any movie review.
* Click **Analyze Sentiment**.
* View the result immediately on the interface.

---
