import unittest
from KNN import KNN
import main
import pandas as pd
import numpy as np

class KNNTest(unittest.TestCase):
    iris_test_filename = 'iris.data.test.csv'
    iris_learning_filename = 'iris.data.learning.csv'

    irisTest = pd.read_csv(iris_test_filename, sep=',', decimal='.', header=None,
                           names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    irisLearning = pd.read_csv(iris_learning_filename, sep=',', decimal='.', header=None,
                               names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    untagged_test_data = irisTest.iloc[:, :-1].values
    test_data_targets = irisTest.iloc[:, 4].values

    def test_score(self):
        K2 = KNN(3, self.irisLearning)
        self.assertEquals(93.33333333333333, K2.score(self.untagged_test_data, self.test_data_targets))

    if __name__ == '__main__':
        unittest.main()