from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.base import clone,BaseEstimator,TransformerMixin,ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt 

class StackingClassifer(BaseEstimator,ClassifierMixin,TransformerMixin):

    def __init__(self,classifiers):
        self.classifiers = classifiers
        self.meta_classifier = DecisionTreeClassifier()

    def fit(self,X,y):
        for clf in self.classifiers:
            clf.fit(X,y)
        self.meta_classifier.fit(self._get_meta_features(X),y)
        return self

    def _get_meta_features(self,X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers])
        print(probas.shape)
        return np.concatenate(probas, axis=1)

    def predict(self,X):
        return self.meta_classifier.predict(self._get_meta_features(X))

    def predict_proba(self,X):
        return self.meta_classifier.predict_proba(self._get_meta_features(X))


X,y = make_classification(
    n_samples=1000,n_features=50,n_informative=30,n_clusters_per_class=3,
    random_state=11)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=11)

accuracies = []


lr = LogisticRegression()
knn_clf = KNeighborsClassifier()

base_classifiers = [lr,knn_clf]
stacking_clf = StackingClassifer(base_classifiers)
stacking_clf.fit(X_train,y_train)
print(stacking_clf.score(X_test,y_test))