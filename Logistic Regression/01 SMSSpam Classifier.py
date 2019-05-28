import sys
import io
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,roc_curve,auc


# Handle with coding problem
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

# Data
df = pd.read_csv('./SMSSpamCollection',delimiter='\t',header=None)
print('Number of spam messages %s'% df[df[0]=='spam'][0].count())
print('Number of ham messages %s'% df[df[0]=='ham'][0].count())

# Model
X = df[1].values
y = df[0].values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Classifier
classifer = LogisticRegression()
classifer.fit(X_train,y_train)

# Predict
y_predictions = classifer.predict(X_test)

# Metric
## F1 score
f1 = f1_score(y_test,y_predictions,pos_label='ham')
print(f1)

## ROC_AUC
y_predictions_proba = classifer.predict_proba(X_test)
false_postive_rate, recall, thresholds = roc_curve(y_test,y_predictions_proba[:,1],pos_label='spam')
roc_auc = auc(false_postive_rate,recall)

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(false_postive_rate,recall,'b',label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()