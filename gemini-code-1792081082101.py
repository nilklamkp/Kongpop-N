import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Iris Flower Prediction App",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Prediction")
st.write("แอปพลิเคชันสำหรับทำนายสายพันธุ์ดอกไอริส (Iris)")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model_path = 'model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        from sklearn.datasets import load_iris
        iris = load_iris()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(iris.data, iris.target)
        joblib.dump(model, model_path)
    return model

model = load_model()

# ใช้ Form เพื่อป้องกันไม่ให้แอปคำนวณซ้ำซ้อนขณะขยับ Slider
with st.form("prediction_form"):
    st.subheader("กรุณากรอกข้อมูลขนาดของดอกไอริส (หน่วย: เซนติเมตร)")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5, 0.1)
    with col2:
        petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2, 0.1)
    
    submit_button = st.form_submit_button(label="🔍 ทำนายสายพันธุ์", type="primary")

if submit_button:
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    )
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)
    
    iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    result_species = iris_classes[prediction]
    
    st.markdown("---")
    st.subheader("ผลลัพธ์การทำนาย (Prediction Result)")
    st.success(f"สายพันธุ์ที่ทำนายได้คือ: **{result_species}**", icon="🎉")
    
    st.write("### ความน่าจะเป็นของแต่ละสายพันธุ์:")
    proba_df = pd.DataFrame(prediction_proba, columns=iris_classes).T
    proba_df.columns = ['Probability']
    st.bar_chart(proba_df)