import streamlit as st


st.write('''
         Compare differnet **Classifiers** on different **Datasets**
         ''')

st.sidebar.selectbox("Select a Dataset",("Iris Dataset","Breast Cancer","Wine Dataset"))
st.sidebar.selectbox("Select a Classifier",("KNN","SVM","Random Forest"))