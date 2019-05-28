from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt 

# Data
data = np.loadtxt('weight_data.txt', dtype=bytes).astype(str)
X_train = data[:,:2].astype('float')
y_train = data[:,2]

# Hyper Parameter
k = 3

# KNN
X_test = np.array([[175, 70]])
distances = np.sqrt(np.sum((X_train-X_test)**2,axis=1))
### Search the nearest k points
nearest_points_indices = np.argsort(distances)[:3]
nearest_points_genders = y_train[nearest_points_indices]
print(nearest_points_genders)

y_predcition = Counter(nearest_points_genders).most_common(1)[0][0]

# Figure
for i, x in enumerate(X_train):
    plt.scatter(x[0],x[1],marker='x' if y_train[i]=='male' else 'o', color='k')
print(X_test)
plt.scatter(X_test[0][0],X_test[0][1],marker='x' if y_predcition=='male' else 'o', color='r')
plt.ylabel('weight')
plt.xlabel('height')
plt.grid(True)
plt.show()