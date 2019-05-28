import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fetch Data
mnist = fetch_mldata('MNIST original', data_home='data/mnist')

# Show parts of Image
# counter = 1
# for i in range(0,10):
#     for j in range(1,11):
#         plt.subplot(10,10,counter)
#         plt.imshow(mnist.data[i*7000+j].reshape(28,28),cmap=plt.cm.gray)
#         plt.axis('off')
#         counter += 1
# plt.show()

# Data
X,y = mnist.data, mnist.target
X = X/255.0 * 2 -1
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=11)
print(y_train)
# SVC
clf = SVC(kernel='rbf',C=3,gamma=0.01)
clf.fit(X_train[:10000],y_train[:10000])

# Prediction
predictions = clf.predict(X_test)

# Metric
print(classification_report(y_test,predictions))