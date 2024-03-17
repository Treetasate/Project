import streamlit as st
import numpy as np
import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import requests
from io import StringIO

# โหลดข้อมูล
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQO5tgsRIpeMbYT0B_b6zTeSZmHkFl9FjPrvCsczYk7-WMgAVj52JpX4Gl72WHp5gLS9hH_g2BG88Ko/pub?gid=2062330787&single=true&output=csv'
df = pd.read_csv(url)

# ปรับปรุงการเตรียมข้อมูล
def prepare_data(df):
    thai_stopwords_list = thai_stopwords()

    def text_process(text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text)
        final = [word for word in tokens if word.lower() not in thai_stopwords_list]
        return " ".join(final)

    df['text_tokens'] = df['บทคัดย่อ'].apply(text_process)

    Xlr = df[['text_tokens']]
    ylr = df['ประเภทของโปรเจค']

    X_train, X_test, y_train, y_test = train_test_split(Xlr, ylr, test_size=0.3)

    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(X_train['text_tokens'].values.astype('U'))

    return cvec, X_train, X_test, y_train, y_test

cvec, X_train, X_test, y_train, y_test = prepare_data(df)

# ฝึกโมเดล
lr = LogisticRegression()
lr.fit(cvec.transform(X_train['text_tokens'].values.astype('U')), y_train)

# ฟังก์ชันทำนาย
def predict_class(input_text):
    processed_input = cvec.transform([input_text])
    prediction = lr.predict(processed_input)
    return prediction[0]

# รับข้อมูลจากผู้ใช้
st.title("🚐 Prediction by LogisticRegression")
input_text = st.text_area("ใส่ข้อความที่นี่", "", height=200)
if st.button("ทำนาย"):
    if not input_text:
        st.write("ไม่มีข้อความ")
    else:
        prediction = predict_class(input_text)
        st.write("ผลการทำนาย:", prediction)
