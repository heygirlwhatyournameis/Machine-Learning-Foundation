import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training data
X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
y = np.array([7,9,13,17.5,18])

# Transform into 3-dimension data
quadratic_featurizer = PolynomialFeatures(degree=2)
X_quadratic = quadratic_featurizer.fit_transform(X)
model = LinearRegression()
model.fit(X_quadratic,y)

# Figure
xx = np.linspace(0,25).reshape(-1,1)
xx_quadratic = quadratic_featurizer.transform(xx)
yy = model.predict(xx_quadratic)
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X,y,'k.')
plt.plot(xx,yy,c='r',linestyle='--')
plt.axis([0,25,0,25])
plt.grid(True)
plt.show()

