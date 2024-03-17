import streamlit as st
import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load data
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQO5tgsRIpeMbYT0B_b6zTeSZmHkFl9FjPrvCsczYk7-WMgAVj52JpX4Gl72WHp5gLS9hH_g2BG88Ko/pub?gid=2062330787&single=true&output=csv'
df = pd.read_csv(url)

# Thai stopwords
thai_stopwords = thai_stopwords()

# Text preprocessing
def text_process(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    final = [word for word in tokens if word.lower() not in thai_stopwords]
    return " ".join(final)

df['text_tokens'] = df['บทคัดย่อ'].apply(text_process)

# Prepare data
X = df['text_tokens']
y = df['ประเภทของโปรเจค']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
X_test_vectorized = tfidf_vectorizer.transform(X_test)

# Training Naive Bayes Model
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, y_train)

# Streamlit UI
st.title("🔮 Prediction by Naivebayes 🔮")
input_text = st.text_area("ใส่ข้อความที่นี่", "",height=200)
if st.button("ทำนาย"):
    if not input_text:
        st.write("กรุณาใส่ข้อความ")
    else:
        processed_text = text_process(input_text)
        X_new = [processed_text]
        X_new_vectorized = tfidf_vectorizer.transform(X_new)
        y_new_pred = naive_bayes_classifier.predict(X_new_vectorized)
        st.write("ผลการทำนาย:", y_new_pred[0])
