import streamlit as st
import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data from Google Spreadsheet
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQO5tgsRIpeMbYT0B_b6zTeSZmHkFl9FjPrvCsczYk7-WMgAVj52JpX4Gl72WHp5gLS9hH_g2BG88Ko/pub?gid=2062330787&single=true&output=csv'
df = pd.read_csv(url)

# Thai stopwords
stopwords = thai_stopwords()

# Text preprocessing function
def text_process(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text, keep_whitespace=False)
    final = [word for word in tokens if word.lower() not in stopwords]
    return " ".join(final)

# Preprocess the text column
df['text_tokens'] = df['บทคัดย่อ'].apply(text_process)

# Train and return a model and vectorizer
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text_tokens'], df['ประเภทของโปรเจค'], test_size=0.3)
    cvec = CountVectorizer(analyzer=lambda x: x.split(' '))
    cvec.fit(X_train)
    lr = LogisticRegression()
    lr.fit(cvec.transform(X_train), y_train)
    return lr, cvec

# Function to predict the class of a new input
def predict_class(input_text, model, vectorizer):
    processed_input = text_process(input_text)
    new_data_bow = vectorizer.transform([processed_input])
    return model.predict(new_data_bow)

# Streamlit UI
st.title(" Prediction by CNN 🖥")
input_text = st.text_area("ใส่ข้อความที่นี่", "", height=200)

# Train the model and prepare the vectorizer
model, vectorizer = train_model(df)

if st.button("ทำนาย"):
    if not input_text:
        st.write("ไม่มีข้อความ")
    else:
        prediction = predict_class(input_text, model, vectorizer)
        st.write("ผลการทำนาย:", prediction)
