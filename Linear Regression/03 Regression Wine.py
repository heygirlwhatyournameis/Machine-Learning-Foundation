import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data
df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train,X_test,y_train,y_test = train_test_split(X,y)

# Find correlations of columns in pair
# df.corr()

# Train
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predict
# R2 Score
y_prediction = regressor.predict(X_test)
print('R-squared: %s' % regressor.score(X_test, y_test))

# My R2 Score
ss_tot = sum( (y_test - np.mean(y_test)) ** 2)
ss_res = sum( (y_test - y_prediction) ** 2)
print('My R-squared: %s' % (1 - ss_res / ss_tot))

# Figure
plt.figure()
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.scatter(df['alcohol'],df['quality'],color='k',marker='.')
plt.title('Alcohol against Quality')
# plt.show()
