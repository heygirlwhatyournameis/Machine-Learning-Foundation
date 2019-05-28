import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training data
X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
y = np.array([7,9,13,17.5,18])

# Estimator
model = LinearRegression()
model.fit(X,y)

# Predict
X_test = np.array([[3],[5],[8],[10],[24]])
y_test = model.predict(X_test)

# Figure
xx = np.linspace(0,25).reshape(-1,1)
yy = model.predict(xx)
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X,y,'k.')
plt.plot(X_test,y_test,'b.')
plt.plot(xx,yy,c='r',linestyle='--')
plt.axis([0,25,0,25])
plt.grid(True)
plt.savefig('01 Linear Regression.png')
plt.show()


