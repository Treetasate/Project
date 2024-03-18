import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split


# Load data
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQO5tgsRIpeMbYT0B_b6zTeSZmHkFl9FjPrvCsczYk7-WMgAVj52JpX4Gl72WHp5gLS9hH_g2BG88Ko/pub?gid=2062330787&single=true&output=csv'
df = pd.read_csv(url)

# Thai stopwords
thai_stopwords_list = thai_stopwords()

# Text preprocessing function
def text_process(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    final = [word for word in tokens if word.lower() not in thai_stopwords_list]
    return " ".join(final)

# Preprocess the dataframe
df['text_tokens'] = df['บทคัดย่อ'].apply(text_process)

def appLSTM():
    # Prepare data for training
    texts = df['text_tokens']
    labels = df['ประเภทของโปรเจค']
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    max_words = 10000
    max_len = 100
    embedding_dim = 100
    lstm_units = 64
    num_classes = len(set(labels))

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=max_len)
    y = keras.utils.to_categorical(labels, num_classes)

    # Build and compile the model
    modellstm = Sequential()
    modellstm.add(Embedding(max_words, embedding_dim))
    modellstm.add(LSTM(lstm_units))
    modellstm.add(Dense(num_classes, activation='softmax'))
    modellstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    modellstm.fit(x, y, batch_size=32, epochs=10)

    # Prediction function
    def predict_class(input_text):
        processed_text = text_process(input_text)
        sequences_new = tokenizer.texts_to_sequences([processed_text])
        sequences_padded_new = pad_sequences(sequences_new, maxlen=max_len)
        y_pred_classes_new = modellstm.predict(sequences_padded_new).argmax(axis=1)
        return label_encoder.inverse_transform(y_pred_classes_new)[0]

    # Streamlit UI
    st.title("🐱‍👓 Prediction by LSTM 🐱‍🏍")
    input_text = st.text_area("ใส่ข้อความที่นี่","",height=200)
    if st.button("ทำนาย"):
        if not input_text:
            st.write("ไม่มีข้อความ")
        else:
            prediction = predict_class(input_text)
            st.write("ผลการทำนาย:", prediction)

def appRandom():
    XRd = df['text_tokens']
    yRd = df['ประเภทของโปรเจค']

    X_train, X_test, y_train, y_test = train_test_split(XRd, yRd, test_size=0.3, random_state=42)

    tfidf_vectorizer = TfidfVectorizer()
    X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
    X_test_vectorized = tfidf_vectorizer.transform(X_test)

    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_classifier.fit(X_train_vectorized, y_train)

    st.title("🎄 Prediction by Random Forest 🌳")

    # รวมช่องข้อความและการทำนายเข้าด้วยกัน
    input_text = st.text_area("ใส่ข้อความที่นี่", "",height=200)
    if st.button("ทำนาย"):
        if not input_text:
            st.write("ไม่มีข้อความ")
        else:
            # ใช้โค้ดทำนายที่คุณมีแล้ว
            # Vectorize the input text using the same TfidfVectorizer
            X_new = [text_process(input_text)]
            X_new_vectorized = tfidf_vectorizer.transform(X_new)
            
            # Predict using the trained Random Forest model
            y_new_pred = random_forest_classifier.predict(X_new_vectorized)

            # แสดงผลลัพธ์
            st.write("ผลการทำนาย:", y_new_pred[0])

def appNavivebayes():
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

def appLogisticRegression():
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

def appCNN():
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

st.title("🔮 Predict machine 🔮")

# ใช้ selectbox เพื่อให้ผู้ใช้เลือกหน้า
page = st.selectbox('เลือกเครื่องมือทำนาย : ', ['LSTM', 'RandomForest', 'Navivebayes', 'LogisticRegression', 'CNN'])

# เรียกใช้ฟังก์ชั่นของหน้าตามการเลือกของผู้ใช้
if page == 'LSTM':
    appLSTM()
elif page == 'RandomForest':
    appRandom()
elif page == 'Navivebayes':
    appNavivebayes()
elif page == 'LogisticRegression':
    appLogisticRegression()
elif page == 'CNN':
    appCNN()