from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# Data
data = np.loadtxt('weight_data.txt', dtype=bytes).astype(str)
X_train = data[:,:2].astype('float')
y_train_raw = data[:,2]
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train_raw)[:,0]

X_test = np.array([[175, 70],[168,65],[180,96],[160,52],[169,67]])
y_test = ['female','male','male','female','female']

# Hyper Parameter
k = 3

# KNN
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train,y_train)

y_predcitions_binary = clf.predict(X_test)
y_predcitions = lb.inverse_transform(y_predcitions_binary)

# Metric
print('Accuracy: %s' % accuracy_score(y_test,y_predcitions))

# Figure
plt.grid(True)
for i, x in enumerate(X_train):
    plt.scatter(x[0],x[1],marker='x' if y_train_raw[i]=='male' else 'o', color='k')

for i, x in enumerate(X_test):
    plt.scatter(x[0],x[1],marker='x' if y_predcitions[i]=='male' else 'o', color='r')


plt.ylabel('weight')
plt.xlabel('height')
plt.show()