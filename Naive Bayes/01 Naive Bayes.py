import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model.logistic import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Dataset 
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=31)

train_sizes = range(10,len(X_train),25)


lr = LogisticRegression()
nb = GaussianNB()

lr_scores = []
nb_scores = []

# Epoches
for train_size in train_sizes:
    X_slice,_,y_slice,_=train_test_split(X_train,y_train,train_size=train_size,stratify=y_train,random_state=31)
    nb.fit(X_slice,y_slice)
    nb_scores.append((nb.score(X_test,y_test)))
    lr.fit(X_slice,y_slice)
    lr_scores.append((lr.score(X_test,y_test)))

# Figure
plt.figure()
plt.title("Naive Bayes and Logistic Regression Accuracies")
plt.xlabel("Number of training instances")
plt.ylabel("Test set accuracy")
plt.grid(True)
plt.plot(train_sizes,nb_scores,label='Naive Bayes')
plt.plot(train_sizes,lr_scores,label='Logistic Regression',linestyle='--')
plt.legend()
plt.savefig('Naive Bayes and Logistic Regression Accuracies.png')
# plt.show()
