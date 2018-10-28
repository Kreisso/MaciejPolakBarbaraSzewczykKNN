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

    k_tab = [8, 20, 5, 12, 3, -2]

    def test_init(self):
        for i in range(len(self.k_tab)):
            self.assertEqual(type(self.k_tab[i]), int)
            self.assertTrue(0 <= self.k_tab[i] <= len(self.unittest_learning_data))
        self.assertEqual(type(self.unittest_learning_data), pd.DataFrame)

    def test_predict(self):
        self.assertEqual(type(self.untagged_test_data), np.ndarray)


    def test_score(self):
        K = []
        for i in range(len(self.k_tab)):
            K.append(KNN(self.k_tab[i], self.unittest_learning_data))
            print(K[i].score(self.untagged_test_data, self.test_data_targets))
            print(K[i].predict(self.untagged_test_data))
            self.assertEqual(type(self.untagged_test_data), np.ndarray)
            self.assertEqual(type(self.test_data_targets), np.ndarray)
            self.assertEqual(75, K[i].score(self.untagged_test_data, self.test_data_targets))

    if __name__ == '__main__':
        unittest.main()