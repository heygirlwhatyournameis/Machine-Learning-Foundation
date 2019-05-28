import numpy as np 
from sklearn.linear_model import Perceptron 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups 
from sklearn.metrics import classification_report 

# Data 
categories = ['rec.sport.hockey','rec.sport.baseball','rec.autos']
newsgroup_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'))
newsgroup_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers','footers','quotes'))

# Features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroup_train.data)
X_test = vectorizer.transform(newsgroup_test.data)
y_train = newsgroup_train.target
y_test = newsgroup_test.target

# Estimator
clf = Perceptron(random_state=11)
clf.fit(X_train,y_train)

# Predict
y_prediction = clf.predict(X_test)

# Metric
print(classification_report(y_test,y_prediction))