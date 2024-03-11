import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('lstm.pkl', 'rb'))

def predict_summary(text):
  X = np.array([text])
  y_pred = model.predict(X)
  return y_pred[0]



st.title("📏 Prediction LSTM 🙏")

input_area = st.text_area("ป้อนข้อความที่นี่")

if st.button("ทำนาย"):
  summary = predict_summary(input_area)
  st.write(summary)

