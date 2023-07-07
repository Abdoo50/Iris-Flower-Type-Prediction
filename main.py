import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()

X = iris.data
y = iris.target
rfc = RandomForestClassifier()

rfc.fit(X, y)

# streamlit structure
st.write("Simple Iris **Flower** Data Science Project!")
st.sidebar.header("Input Header")

def User_Inputs():
    first = st.sidebar.slider('Sepal_Length', 4.3, 7.9, 5.4)
    second = st.sidebar.slider('Sepal_Width', 2.5, 4.4, 3.4)
    third = st.sidebar.slider('Petal_Length', 1.0, 6.9, 1.3)
    fourth = st.sidebar.slider('Petal_Width', 0.1, 2.5, 0.2)

    data = {
        'Sepal_Length': first,
        'Sepal_Width': second,
        'Petal_Length': third,
        'Petal_Width': fourth
    }
    Features = pd.DataFrame(data, index = [0])

    return Features

df = User_Inputs()

st.subheader('User Inputs Parameters')
st.write(df)

st.subheader('Target Names')
st.write(iris.target_names)

predict = rfc.predict(df)
proba = rfc.predict_proba(df)

st.subheader('Prediction')
st.write(iris.target_names[predict])

st.subheader('Prediction Probability')
st.write(proba)