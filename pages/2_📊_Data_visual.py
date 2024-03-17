import streamlit as st
import pandas as pd
import plotly.express as px

# โหลดข้อมูลจากไฟล์ CSV
url = './Data/Thesis.csv'
df = pd.read_csv(url)

# กำหนดหัวข้อของเว็บแอป
st.title("📉 Dashboard from CED 📈")

col1, col2 = st.columns(2)

with col1:
    selected_years = st.multiselect("เลือกปีการศึกษา:", options=df['ปีการศึกษา'].unique(), default=df['ปีการศึกษา'].unique())

with col2:
    selected_types = st.multiselect("เลือกประเภทของโปรเจค:", options=df['ประเภทของโปรเจค'].unique(), default=df['ประเภทของโปรเจค'].unique())

selected_advisors = st.multiselect("เลือกอาจารย์ที่ปรึกษา:", options=df['อาจารย์ที่ปรึกษา'].unique(), default=df['อาจารย์ที่ปรึกษา'].unique())

# กรองข้อมูลตามที่เลือก
filtered_data = df[
    df['ปีการศึกษา'].isin(selected_years) & 
    df['ประเภทของโปรเจค'].isin(selected_types) & 
    df['อาจารย์ที่ปรึกษา'].isin(selected_advisors)
]

# ตรวจสอบว่ามีข้อมูลหลังจากกรองหรือไม่
if not filtered_data.empty:
    
    df['ปีการศึกษา'] = df['ปีการศึกษา'].astype(str)
    project_type_counts = filtered_data['ประเภทของโปรเจค'].value_counts().reset_index()
    project_type_counts.columns = ['ประเภทของโปรเจค', 'counts']

    st.subheader("แผนภูมิแท่ง")
    fig_bar = px.bar(filtered_data, x='ปีการศึกษา', y='ประเภทของโปรเจค', color='อาจารย์ที่ปรึกษา')
    fig_bar.update_layout(xaxis_tickformat='d')
    st.plotly_chart(fig_bar)

    st.subheader("แผนภูมิวงกลม")
    fig_pie = px.pie(project_type_counts, names='ประเภทของโปรเจค', values='counts')
    st.plotly_chart(fig_pie)

    st.subheader("กราฟ")
    fig = px.line(filtered_data, x='ปีการศึกษา', y='อาจารย์ที่ปรึกษา', color='ประเภทของโปรเจค')
    st.plotly_chart(fig)

else:
    st.write("กรุณาเลือกข้อมูลเพื่อแสดงแผนภูมิ")