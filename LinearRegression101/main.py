# Imports
import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("LinearRegression101\student-mat.csv", sep=";")
print(data.head())
data = data [["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
acc = linear.score(X_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# predictions = linear.predict(X_test)
# for x in range(len(predictions)):
#     print(f'Predicted: {predictions[x]}, Data: {X_test[x]}, Actual: {y_test[x]}')   