#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:37:37 2021

@author: yiranliu
"""
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("/Users/yiranliu/Desktop/HW3/classifier.pkl","rb")
classifier=pickle.load(pickle_in)


    
#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])


def predict_note_authentication(test):
    mydf=pd.read_csv("/Users/yiranliu/Desktop/HW3//Hotel_reviews.csv")
    X_data=mydf['Review']
    y_labels=mydf['Authenticity']
    X_data_train,X_data_test,y_labels_train,y_labels_test=train_test_split(X_data,y_labels,test_size=0.2,random_state=1)
    
    tfidf_vectorizer=TfidfVectorizer()
    train_tfidf=tfidf_vectorizer.fit_transform(X_data_train)

    test_tfidf= tfidf_vectorizer.transform([test])
   
    prediction=classifier.predict(test_tfidf)
    print(prediction)    
     
    return prediction[0]


def lime(review):
    mydf=pd.read_csv("/Users/yiranliu/Desktop/HW3//Hotel_reviews.csv")
    X_data=mydf['Review']
    y_labels=mydf['Authenticity']
    X_data_train,X_data_test,y_labels_train,y_labels_test=train_test_split(X_data,y_labels,test_size=0.2,random_state=1)
    
    tfidf_vectorizer=TfidfVectorizer()
    train_tfidf=tfidf_vectorizer.fit_transform(X_data_train)

    test_tfidf= tfidf_vectorizer.transform([review])
    c = make_pipeline(tfidf_vectorizer, classifier)
    explainer = LimeTextExplainer(class_names=['deceptive','truthful'])
    exp = explainer.explain_instance(review, c.predict_proba, num_features=6)
    components.html(exp.as_html(), height=800)
    
       

def main():
    st.title("Opinion Spam Detection Application")
    html_temp = """
    <div>
    <h3>Please input the hotel review:</h3>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    review = st.text_input("")
    
    result=""

    if st.button("Submit"):
        result=predict_note_authentication(review)
        
        st.success('The authenticity of this hotel review is: {}'.format(result))
        lime(review)
        
       

if __name__=='__main__':
    main()