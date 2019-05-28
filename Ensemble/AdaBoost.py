from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

X,y = make_classification(
    n_samples=1000,n_features=50,n_informative=30,n_clusters_per_class=3,
    random_state=11)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=11)

accuracies = []

for i in range(1,51):
    clf = AdaBoostClassifier(n_estimators = i,random_state=11)
    clf.fit(X_train,y_train)
    accuracies.append(clf.score(X_test,y_test))

plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimator in ensemble')
plt.grid(True)
plt.plot(range(1,51),accuracies)
plt.show()