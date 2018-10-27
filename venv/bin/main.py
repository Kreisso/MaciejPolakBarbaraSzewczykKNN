import pandas as pd
import numpy as np

from KNN import KNN

iris_test_filename = 'iris.data.test.csv'
iris_learning_filename = 'iris.data.learning.csv'

irisTest = pd.read_csv(iris_test_filename, sep=',', decimal='.', header=None, names= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','target'])
irisLearning = pd.read_csv(iris_learning_filename, sep=',', decimal='.', header=None, names= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','target'])
unTagList = irisTest.iloc[:, :-1].values
testTargets = irisTest.iloc[:,4].values
k = 3

K1 = KNN(k,irisLearning)

recognized = K1.predict(unTagList)

score = K1.score(testTargets, recognized)
print(type(testTargets))
print(type(recognized))
print(score)
