# Imports
import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle


data = pd.read_csv(r"C:\Users\marie\Projets\ML-FromScratch\LinearRegression101\student-mat.csv", sep=";")
print(data.head())
data = data [["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

best=0
for _ in range(20):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    acc = linear.score(X_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# predictions = linear.predict(X_test)
# for x in range(len(predictions)):
#     print(f'Predicted: {predictions[x]}, Data: {X_test[x]}, Actual: {y_test[x]}')   

# Plotting
p = "absences"
m = "G3"
style.use("ggplot")
pyplot.scatter(data[p], data[m])
pyplot.xlabel(p)
pyplot.ylabel(m)
pyplot.show()