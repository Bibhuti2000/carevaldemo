import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']]


# Conversion of data
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])

print(X)

# label mapping
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)


#split the data into training an testing part
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# creation of model
knn = neighbors.KNeighborsClassifier(n_neighbors=25,weights="distance")

#train model using  fit function

knn.fit(X_train,y_train)

#make predictions on test data after training.

predictions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)

print("predictions : ", predictions)
print("accuracy : ",accuracy)


print("Actual value : ", y[:])
print("prediction_value : ",knn.predict(X)[:])


def predictionReq(arr):
    N = [[arr[0],arr[0],arr[0]]]
    return knn.predict(N)


