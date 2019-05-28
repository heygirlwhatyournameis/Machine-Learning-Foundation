import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Data
c1x = np.random.uniform(0.5, 1.5, (1,10))
c1y = np.random.uniform(0.5, 1.5, (1,10))
c2x = np.random.uniform(3.5, 4.5, (1,10))
c2y = np.random.uniform(3.5, 4.5, (1,10))

x = np.hstack((c1x,c2x))
y = np.hstack((c1y,c2y))
X = np.vstack((x,y)).T

K = range(1,10)
meanDispersions = []


# KMeans
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    print(np.min(cdist(X,kmeans.cluster_centers_),axis=1))
    meanDispersions.append(sum(np.min(cdist(X,kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])


# Figure
plt.figure()
plt.xlabel('Number of clusters')
plt.ylabel('')
plt.title('Elbow Method')
plt.plot(K,meanDispersions,'bx-')
plt.show()