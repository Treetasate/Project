import streamlit as st
import pandas as pd
from functools import reduce 
import plotly.express as px
import numpy as np

file_path = "Data/Thesis.csv"

def load_data():
    data = pd.read_csv(file_path)
    data.fillna("", inplace=True)
    data = data.astype(str)
    return data

data = load_data()

filtered_data = data  # Replace with your actual filtered data logic

# Set page to wide mode
st.set_page_config(layout="wide")

# Custom CSS to expand the dataframe
st.markdown("""
<style>
    .dataframe-container {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Pagination setup
page_size = 10  # Adjust as needed
total_pages = len(filtered_data) // page_size + (1 if len(filtered_data) % page_size > 0 else 0)
current_page = st.session_state.get('current_page', 0)

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
    st.markdown(f"อาจารย์ทั้งหมด : {filtered_data['อาจารย์ที่ปรึกษา'].nunique()}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"โปรเจคทั้งหมด : {filtered_data.shape[0]}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"นักศึกษาทั้งหมด : {filtered_data['ชื่อผู้ทำ1'].nunique()}</div>", unsafe_allow_html=True)

st.title("")
# Displaying the data for the current page
start_row = current_page * page_size
end_row = start_row + page_size
st.dataframe(filtered_data.iloc[start_row:end_row])

col1, col2, col3, col4 = st.columns([3,1,1,3])
with col2:
    if st.button('Previous'):
        if current_page > 0:
            current_page -= 1
            st.session_state['current_page'] = current_page
with col3:
    if st.button('Next'):
        if current_page + 1 < total_pages:
            current_page += 1
            st.session_state['current_page'] = current_page

# Optionally, display current page info
col1, col2, col3, col4, col5 = st.columns([3,3,2,3,3])  # Adjust the ratio as needed for better centering
with col3: 
    st.write(f"Page {current_page + 1} of {total_pages}")

st.header("")
# Assuming 'filtered_data' is already defined and correctly filtered elsewhere in your script

# Group data to find the advisor with the most projects per project type
project_counts = filtered_data.groupby(['ประเภทของโปรเจค', 'อาจารย์ที่ปรึกษา']).size().reset_index(name='จำนวนโปรเจค')

# Handle NaN values and convert 'จำนวนโปรเจค' to int
project_counts['จำนวนโปรเจค'] = project_counts['จำนวนโปรเจค'].fillna(0).astype(int)

# Find the advisor with the maximum projects per project type
max_projects_advisors = project_counts.groupby('ประเภทของโปรเจค').apply(
    lambda x: x.loc[x['จำนวนโปรเจค'].idxmax()]
).reset_index(drop=True)

# Convert DataFrame to a dictionary and back to DataFrame to ensure all types are JSON serializable
data_dict = max_projects_advisors.to_dict(orient='records')
clean_df = pd.DataFrame(data_dict)

# Ensure types are native Python types for JSON serialization
clean_df['จำนวนโปรเจค'] = clean_df['จำนวนโปรเจค'].astype(int)

# Display the DataFrame
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.write("Advisor with the Most Projects by Project Type")
    st.dataframe(clean_df[['ประเภทของโปรเจค', 'อาจารย์ที่ปรึกษา', 'จำนวนโปรเจค']])

project_type_counts = filtered_data['ประเภทของโปรเจค'].value_counts().reset_index()
project_type_counts.columns = ['ประเภทของโปรเจค', 'Count']
fig_bar = px.bar(project_type_counts, x='ประเภทของโปรเจค', y='Count', title='Distribution of Project Types')
st.plotly_chart(fig_bar)


advisor_counts = filtered_data['อาจารย์ที่ปรึกษา'].value_counts().reset_index()
advisor_counts.columns = ['อาจารย์ที่ปรึกษา', 'Count']
fig_pie = px.pie(advisor_counts, values='Count', names='อาจารย์ที่ปรึกษา', title='Distribution of Projects by Advisor')
st.plotly_chart(fig_pie)
