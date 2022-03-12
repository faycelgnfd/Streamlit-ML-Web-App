import streamlit as st
import numpy as np
from sklearn import datasets

#models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



st.write('''
         Compare differnet **Classifiers** on different **Datasets**
         ''')

dataChoice = st.sidebar.selectbox("Select a Dataset",("Iris Dataset","Breast Cancer","Wine Dataset"))
classifier = st.sidebar.selectbox("Select a Classifier",("KNN","SVM","Random Forest"))

#function to load the chosen dataset
def get_dataset(dataChoice):
    if dataChoice == "Iris Dataset":
        dataset = datasets.load_iris()
    elif dataChoice == "Breast Cancer":
        dataset = datasets.load_breast_cancer()
    else:
        dataset = datasets.load_wine()
    
    return dataset.data, dataset.target

#getting infos of the chosen dataset
X, y = get_dataset(dataChoice)
st.write("Number of samples",X.shape[0])
st.write("Number of features",X.shape[1])
st.write("Number of classes",len(np.unique(y)))

#set UI according to the chosen classifier and setting its parameters
def add_parameters_ui(classifier):
    params = dict()
    if classifier == "KNN":
        K = st.sidebar.slider("K neighbours",1,15)
        params["K"] = K
    elif classifier == "SVM":
        C = st.sidebar.slider("C value",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth",2,15)
        estimators = st.sidebar.slider("Num of estimators",1,100)
        params["max_depth"] = max_depth
        params["estimators"] = estimators
    return params

params = add_parameters_ui(classifier)

#function de set classifier
def get_classifier(classifier, params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["estimators"], max_depth=params["max_depth"])
    return clf

clf = get_classifier(classifier, params)
st.write("Chosen Classifier : ",clf)
        