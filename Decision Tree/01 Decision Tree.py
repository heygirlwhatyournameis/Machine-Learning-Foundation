import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

# Data
df = pd.read_csv('ad.data',delimiter=',',header=None)

enable_columns = set(df.columns.values) 
enable_columns.remove(len(df.columns.values)-1)

response_columns = df[len(df.columns.values)-1]

X = df[list(enable_columns)].copy()
y = [1 if e=='ad.' else 0 for e in response_columns]
X.replace(to_replace=' *\?',value='-1',regex=True,inplace=True)

X_train,X_test,y_train,y_test = train_test_split(X,y)

# Estimator
clf = DecisionTreeClassifier(max_depth=150,min_samples_leaf=1,min_impurity_split=3)
clf.fit(X_train,y_train)

# Predcition
predictions = clf.predict(X_test)
print(classification_report(y_test,predictions))
'''
'''