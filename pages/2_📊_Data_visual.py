import streamlit as st
import pandas as pd
from functools import reduce 
import plotly.express as px

file_path = "Data/Thesis.csv"

def load_data():
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    data.fillna("", inplace=True)
    data = data.astype(str)
    return data

data = load_data()

# Custom CSS for dashboard styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:bold;
}
.red-font {
    color:white;
}
.blue-font {
    color:white;
}
.green-font {
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("📉 Dashboard from CED 📈")

# Selection boxes for filtering
col1, col2, col3 = st.columns(3)
with col1:
    selected_advisor = st.selectbox("เลือกอาจารย์ที่ปรึกษา", ["ทั้งหมด"] + list(data['อาจารย์ที่ปรึกษา'].unique()))

with col2:
    selected_type = st.selectbox("เลือกประเภทของโปรเจค", ["ทั้งหมด"] + list(data['ประเภทของโปรเจค'].unique()))

with col3:
    # Extract unique years, sort them, and include "ทั้งหมด"
    unique_years = sorted(data['ปีการศึกษา'].unique(), key=lambda x: int(x))
    selected_year = st.selectbox("เลือกปีการศึกษา", ["ทั้งหมด"] + unique_years)

# Apply filtering based on selection
conditions = []
if selected_year != "ทั้งหมด":
    conditions.append(data['ปีการศึกษา'] == selected_year)
if selected_type != "ทั้งหมด":
    conditions.append(data['ประเภทของโปรเจค'] == selected_type)
if selected_advisor != "ทั้งหมด":
    conditions.append(data['อาจารย์ที่ปรึกษา'] == selected_advisor)

if conditions:
    filtered_data = data.loc[reduce(lambda x, y: x & y, conditions)]
else:
    filtered_data = data.copy()

# Recalculate totals based on filtered data
unique_projects = filtered_data.shape[0]
unique_authors = pd.concat([filtered_data[col] for col in ['ชื่อผู้ทำ1', 'ชื่อผู้ทำ2', 'ชื่อผู้ทำ3'] if col in filtered_data]).nunique()
unique_advisors = pd.concat([filtered_data[col] for col in ['อาจารย์ที่ปรึกษา', 'ที่ปรึกษาร่วม1', 'ที่ปรึกษาร่วม2'] if col in filtered_data]).nunique()

# Displaying totals
st.title("")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='big-font red-font'>โปรเจคทั้งหมด : {unique_projects}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='big-font blue-font'>นักศึกษาทั้งหมด : {unique_authors}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='big-font green-font'>อาจารย์ทั้งหมด : {unique_advisors}</div>", unsafe_allow_html=True)

st.title("")
st.dataframe(filtered_data)  # You can use st.table(filtered_data) if no interactivity is needed

# Creating a bar chart for the project types
project_type_counts = filtered_data['ประเภทของโปรเจค'].value_counts().reset_index()
project_type_counts.columns = ['ประเภทของโปรเจค', 'Count']
fig_bar = px.bar(project_type_counts, x='ประเภทของโปรเจค', y='Count', title='Distribution of Project Types')
st.plotly_chart(fig_bar)

# Creating a pie chart for the advisors
advisor_counts = filtered_data['อาจารย์ที่ปรึกษา'].value_counts().reset_index()
advisor_counts.columns = ['อาจารย์ที่ปรึกษา', 'Count']
fig_pie = px.pie(advisor_counts, values='Count', names='อาจารย์ที่ปรึกษา', title='Distribution of Projects by Advisor')
st.plotly_chart(fig_pie)
