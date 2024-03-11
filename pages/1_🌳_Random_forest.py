import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

st.title('📏 Prediction Random forest 🙏')
