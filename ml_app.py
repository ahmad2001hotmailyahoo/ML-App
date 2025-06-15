import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write('''
# Explore Different ML models and datasets.
Let see how they all works.
''')

dataset_name = st.sidebar.selectbox("Select DataSet", ('Iris', 'Breast Cancer', 'Wine'))
classifier_name = st.sidebar.selectbox("Select DataSet", ['KNN', 'SVM', 'Random Forest'])

def get_dataset():
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target

    return x, y

X, y = get_dataset()

st.write("Shape of datasets:", X.shape)
st.write("Number of classes:", len(np.unique(y)))


def add_parameter_url(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_dept = st.sidebar.slider('max_dept', 2, 15)
        params['max_dept'] = max_dept
        n_estimators = st.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_url(classifier_name)


def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                     max_depth=params['max_dept'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = #{classifier_name}')
st.write(f'Accuracy = #{acc}')

pca = PCA(2)
x_projected = pca.fit_transform(X)

x_1 = x_projected[:,0]
x_2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x_1, x_2, c= y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)
