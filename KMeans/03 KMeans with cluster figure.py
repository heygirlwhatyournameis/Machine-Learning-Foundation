import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

# Data
c1x = np.random.uniform(0.5, 1.5, (1,10))
c1y = np.random.uniform(0.5, 1.5, (1,10))
c2x = np.random.uniform(3.5, 4.5, (1,10))
c2y = np.random.uniform(3.5, 4.5, (1,10))

x = np.hstack((c1x,c2x))
y = np.hstack((c1y,c2y))
X = np.vstack((x,y)).T

k = 2


kmeans = KMeans(n_clusters=k)
result = kmeans.fit_predict(X)

# Silhouette Coefficient
score=silhouette_score(X,kmeans.labels_)

# Figure
plt.figure()
plt.scatter(X[:,0],X[:,1],c=result,cmap='Set1')
plt.title('K = %s, Silhouette Score= %.03f' % (k,score))
plt.show()