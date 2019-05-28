import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neural_network import MLPClassifier

# Data
X_train = np.array([
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1]
])
y_train = [0,1,1,0]

# Estimator
clf = MLPClassifier(hidden_layer_sizes=(3,),activation='logistic',solver='lbfgs',random_state=20)
clf.fit(X_train,y_train)

# Figure
xx,yy = np.meshgrid(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
X_test = np.c_[xx.ravel(),yy.ravel()]
Z = clf.predict(X_test)
Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,cmap='Set2')
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.show()