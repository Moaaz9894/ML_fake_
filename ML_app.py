"""
Created on Sun Apr 30 21:39:50 2023

@author: Moaaz Elsadany
"""
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier





# loading the saved models

news_dataset = pd.read_csv("train.csv")
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')
news_dataset = news_dataset.drop(["title", "author"], axis = 1)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


news_dataset["text"] = news_dataset["text"].apply(wordopt)
X = news_dataset['text']
Y = news_dataset['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(X_train)
xv_test = vectorizer.transform(X_test)

log_model= LogisticRegression()
#log_model.fit(xv_train, Y_train)

ada_model= AdaBoostClassifier()
#ada_model.fit(xv_train, Y_train)

dt_model= DecisionTreeClassifier()
#dt_model.fit(xv_train, Y_train)


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"
    
def log_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_LR = log_model.predict(new_xv_test)
    
    return (output_lable(pred_LR[0]))  


def dt_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_LR = dt_model.predict(new_xv_test)
    
    return (output_lable(pred_LR[0])) 


def ada_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_LR = ada_model.predict(new_xv_test)
    
    return (output_lable(pred_LR[0])) 
  
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Diabetes Prediction Web App',
                          
                          ['AdaBoost',
                           'logistic Regression',
                           'DecisionTree'],
                          icons=['Logistic Regression','SVM','DT'],
                          default_index=0)




    
# Diabetes Prediction Page
if (selected == 'AdaBoost'):
    
    # page title
    st.title('AIE121 - AdaBoost')
    
   # getting the input data from the user
    col1, col2, col3 = st.columns(3)
   
    with col1:
       news_text = st.text_input('Enter the news')
       

   
    # code for Prediction
    ada_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('AdaBoost Regression Test Result'):
        ada_diagnosis = ada_testing(news_text)
     

        
    st.success(ada_diagnosis)


# Diabetes Prediction Page
if (selected == 'logistic Regression'):
    
    # page title
    st.title('AIE121 - logistic Regression')
    
   # getting the input data from the user
    col1, col2, col3 = st.columns(3)
   
    with col1:
       news_text = st.text_input('Enter the news')
       

   
    # code for Prediction
    log_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('logistic Regression Test Result'):
        log_diagnosis = log_testing(news_text)
     

        
    st.success(log_diagnosis)

# Diabetes Prediction Page
if (selected == 'DecisionTree'):
    
    # page title
    st.title('AIE121 - DecisionTree')
    
   # getting the input data from the user
    col1, col2, col3 = st.columns(3)
   
    with col1:
       news_text = st.text_input('Enter the news')
       

   
    # code for Prediction
    dt_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('DecisionTree Test Result'):
        dt_diagnosis = dt_testing(news_text)

        
    st.success(dt_diagnosis)
