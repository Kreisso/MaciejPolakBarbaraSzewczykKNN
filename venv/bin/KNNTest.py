import unittest
from KNN import KNN
import pandas as pd
import numpy as np

class KNNTest(unittest.TestCase):
    unittest_test_filename = 'unittest.data.test.csv'
    unittest_learning_filename = 'unittest.data.learning.csv'

    unittest_test_data = pd.read_csv(unittest_test_filename, sep=',', decimal='.', header=None,
                           names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    unittest_learning_data = pd.read_csv(unittest_learning_filename, sep=',', decimal='.', header=None,
                               names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    untagged_test_data = unittest_test_data.iloc[:, :-1].values
    test_data_targets = unittest_test_data.iloc[:, 4].values

    def test_score(self):
        K2 = KNN(5, self.unittest_learning_data)
        print(K2.score(self.untagged_test_data, self.test_data_targets))
        print(K2.predict(self.untagged_test_data))
        self.assertEqual(75, K2.score(self.untagged_test_data, self.test_data_targets))

    if __name__ == '__main__':
        unittest.main()