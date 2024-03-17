import streamlit as st
import numpy as np
import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

# Load data from Google Spreadsheet
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQO5tgsRIpeMbYT0B_b6zTeSZmHkFl9FjPrvCsczYk7-WMgAVj52JpX4Gl72WHp5gLS9hH_g2BG88Ko/pub?gid=2062330787&single=true&output=csv'
df = pd.read_csv(url)

# Thai stopwords
thai_stopwords = thai_stopwords()

# Text preprocessing function
def text_process(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    final = [word for word in tokens if word.lower() not in thai_stopwords]
    return " ".join(final)

# Apply text preprocessing to 'บทคัดย่อ' column
df['text_tokens'] = df['บทคัดย่อ'].apply(text_process)

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
def LSTM():
    st.title("🐱‍👓 Prediction by LSTM 🐱‍🏍")
    input_text = st.text_area("ใส่ข้อความที่นี่","",height=200)
    if st.button("ทำนาย"):
        if not input_text:
            st.write("ไม่มีข้อความ")
        else:
            prediction = predict_class(input_text)
            st.write("ผลการทำนาย:", prediction)
