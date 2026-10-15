import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Iris Flower Prediction App",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Prediction")
st.write("แอปพลิเคชันสำหรับทำนายสายพันธุ์ดอกไอริส (Iris) โดยใช้โมเดล Machine Learning (Random Forest)")

# จำลองการสร้างหรือโหลดโมเดล (จากข้อมูลที่คุณให้มา เป็น RandomForestClassifier)
@st.cache_resource
def load_model():
    # หากท่านมีไฟล์โมเดลที่บันทึกไว้ (เช่น model.joblib) สามารถใช้ joblib.load('model.joblib') ได้
    # ในกรณีนี้เราจะสร้างโมเดลจำลองตามโครงสร้างที่ท่านให้มาหรือโหลดหากมีไฟล์
    model_path = 'model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Fallback สร้างโมเดลจำลองมาตรฐานของ Iris ถ้ายังไม่ได้เซฟไฟล์โมเดลแยก
        from sklearn.datasets import load_iris
        iris = load_iris()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(iris.data, iris.target)
        joblib.dump(model, model_path)
    return model

model = load_model()

# ส่วนรับค่าจากผู้ใช้งาน (Sidebar หรือ Main Form)
st.subheader("กรุณากรอกข้อมูลขนาดของดอกไอริส (หน่วย: เซนติเมตร)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("ความยาวกลีบเลี้ยง (Sepal Length)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
    sepal_width = st.number_input("ความกว้างกลีบเลี้ยง (Sepal Width)", min_value=2.0, max_value=4.5, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("ความยาวกลีบดอก (Petal Length)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
    petal_width = st.number_input("ความกว้างกลีบดอก (Petal Width)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# ปุ่มทำนายผล
if st.button("🔍 ทำนายสายพันธุ์", type="primary"):
    # แปลงข้อมูลเป็น DataFrame ตามชื่อฟีเจอร์ที่โมเดลต้องการ[cite: 1]
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    
    # ทำนายผล
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)
    
    # รายชื่อสายพันธุ์ดอกไอริส
    iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    result_species = iris_classes[prediction]
    
    st.markdown("---")
    st.subheader("ผลลัพธ์การทำนาย (Prediction Result)")
    st.success(, icon="🎉")
    
    # แสดงความน่าจะเป็น (Probability)
    st.write("### ความน่าจะเป็นของแต่ละสายพันธุ์:")
    proba_df = pd.DataFrame(prediction_proba, columns=iris_classes).T
    proba_df.columns = ['Probability']
    st.bar_chart(proba_df)