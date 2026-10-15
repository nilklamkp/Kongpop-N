import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load('iris_model.pkl')

model = load_model()

st.title('Iris Flower Classifier')

sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width  = st.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width  = st.slider('Petal Width', 0.1, 2.5, 1.0)

if st.button('Predict'):
    pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    names = ['setosa', 'versicolor', 'virginica']
    st.success(f'Predicted: {names[pred[0]]}')