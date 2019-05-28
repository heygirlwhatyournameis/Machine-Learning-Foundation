from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Data
data = np.loadtxt('weight_data.txt', dtype=bytes).astype(str)
data_height = data[:,0].astype('float')
data_gender_raw = data[:,2]

encoder = LabelEncoder()
data_gender = encoder.fit_transform(data_gender_raw)
X_train_raw = np.array([data_height,data_gender]).T

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
y_train = data[:,1].astype('float')

print(y_train)

# Hyper Parameter
k = 3

# KNN Regressor
reg = KNeighborsRegressor(n_neighbors=k)
reg.fit(X_train,y_train)

# Metric


# Figure
xx = np.arange(150,180,1)
xx_male = np.vstack((xx,np.ones(len(xx)))).T
xx_male_scaled = scaler.transform(xx_male)
yy_male = reg.predict(xx_male_scaled)

xx_female = np.vstack((xx,np.zeros(len(xx)))).T
xx_female_scaled = scaler.transform(xx_female)
yy_female = reg.predict(xx_female_scaled)

plt.plot(xx_male[:,0],yy_male,'b.--',label='male')
plt.plot(xx_female[:,0],yy_female,'r.-',label='female')
plt.xlabel('Height')
plt.ylabel('Weight')

plt.title('Weight against Height of Males and Females')
plt.legend()
plt.grid(True)

plt.savefig('KNN Regression.png')

plt.show()


