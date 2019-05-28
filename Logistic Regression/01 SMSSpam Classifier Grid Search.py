import sys
import io
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV



# Handle with coding problem
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


# Pipeline
pipeline = Pipeline([
    ('vect',TfidfVectorizer(stop_words='english')),
    ('clf',LogisticRegression())
])

parameters = {
    'vect__max_df':(0.25,0.5,0.75),
    'vect__max_features':(2500,5000,10000,None),
    'vect__ngram_range':((1,1),(1,2)),
    'vect__use_idf':(True,False),
    'clf__penalty':('l1','l2'),
    'clf__C':(0.01,0.1,1,10),
}


# Data
df = pd.read_csv('./SMSSpamCollection',delimiter='\t',header=None)
print('Number of spam messages %s'% df[df[0]=='spam'][0].count())
print('Number of ham messages %s'% df[df[0]=='ham'][0].count())

# Model
X = df[1].values
y = df[0].values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,y)

# Search
grid_search = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='accuracy',cv=3)
grid_search.fit(X_train_raw,y_train)
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print('t%s: %r' % (param_name,best_parameters[param_name]))
predictions = grid_search.predict(X_test_raw)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Precision:',precision_score(y_test,predictions,pos_label='spam'))
print('Recall:',recall_score(y_test,predictions,pos_label='spam'))

    # precision_score()