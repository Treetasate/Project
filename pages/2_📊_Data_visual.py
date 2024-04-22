import streamlit as st
import pandas as pd
from functools import reduce 
import plotly.express as px

file_path = "Data/Thesis.csv"

def load_data():
    data = pd.read_csv(file_path)
    data.fillna("", inplace=True)
    data = data.astype(str)
    return data

data = load_data()

# Initialize session state for checkboxes if not already done
for advisor in data['อาจารย์ที่ปรึกษา'].unique():
    if advisor not in st.session_state:
        st.session_state[advisor] = True

for project_type in data['ประเภทของโปรเจค'].unique():
    if project_type not in st.session_state:
        st.session_state[project_type] = True

for year in data['ปีการศึกษา'].unique():
    if year not in st.session_state:
        st.session_state[year] = True

st.title("📉 Dashboard from CED 📈")

st.header("")
col1, col2, col3 = st.columns(3)

# Advisor filter
with col1:
    advisor_expander = st.expander("เลือกอาจารย์ที่ปรึกษา")
    selected_advisors = []
    if advisor_expander.button("เลือกทั้งหมด", key='select_all_advisors'):
        for advisor in data['อาจารย์ที่ปรึกษา'].unique():
            st.session_state[advisor] = True
    if advisor_expander.button("ล้างการเลือก", key='reset_advisors'):
        for advisor in data['อาจารย์ที่ปรึกษา'].unique():
            st.session_state[advisor] = False
    for advisor in data['อาจารย์ที่ปรึกษา'].unique():
        is_checked = advisor_expander.checkbox(advisor, value=st.session_state[advisor])
        st.session_state[advisor] = is_checked
        if is_checked:
            selected_advisors.append(advisor)

# Project type filter
with col2:
    type_expander = st.expander("เลือกประเภทของโปรเจค")
    selected_types = []
    if type_expander.button("เลือกทั้งหมด", key='select_all_types'):
        for project_type in data['ประเภทของโปรเจค'].unique():
            st.session_state[project_type] = True
    if type_expander.button("ล้างการเลือก", key='reset_types'):
        for project_type in data['ประเภทของโปรเจค'].unique():
            st.session_state[project_type] = False
    for project_type in data['ประเภทของโปรเจค'].unique():
        is_checked = type_expander.checkbox(project_type, value=st.session_state[project_type])
        st.session_state[project_type] = is_checked
        if is_checked:
            selected_types.append(project_type)

# Year filter
with col3:
    year_expander = st.expander("เลือกปีการศึกษา")
    selected_years = []
    if year_expander.button("เลือกทั้งหมด", key='select_all_years'):
        for year in sorted(data['ปีการศึกษา'].unique(), key=lambda x: int(x)):
            st.session_state[year] = True
    if year_expander.button("ล้างการเลือก", key='reset_years'):
        for year in sorted(data['ปีการศึกษา'].unique(), key=lambda x: int(x)):
            st.session_state[year] = False
    for year in sorted(data['ปีการศึกษา'].unique(), key=lambda x: int(x)):
        is_checked = year_expander.checkbox(year, value=st.session_state[year])
        st.session_state[year] = is_checked
        if is_checked:
            selected_years.append(year)

# Apply filtering based on selection
conditions = []
if selected_advisors:
    conditions.append(data['อาจารย์ที่ปรึกษา'].isin(selected_advisors))
if selected_types:
    conditions.append(data['ประเภทของโปรเจค'].isin(selected_types))
if selected_years:
    conditions.append(data['ปีการศึกษา'].isin(selected_years))

if conditions:
    filtered_data = data.loc[reduce(lambda x, y: x & y, conditions)]
else:
    filtered_data = data.copy()

# Displaying totals
st.header("")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='big-font green-font'>อาจารย์ทั้งหมด : {filtered_data['อาจารย์ที่ปรึกษา'].nunique()}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='big-font red-font'>โปรเจคทั้งหมด : {filtered_data.shape[0]}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='big-font blue-font'>นักศึกษาทั้งหมด : {filtered_data['ชื่อผู้ทำ1'].nunique()}</div>", unsafe_allow_html=True)

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
