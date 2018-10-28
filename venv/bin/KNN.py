from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import euclidean
import sys
import numpy as np
import scipy as sp

class KNN:
    def __init__(self, k, learning_data):
        try:
            if k > len(learning_data):
                raise RuntimeError('K is bigger than count of data learning')
        except Exception as e:
            print(e)
            sys.exit(0)

            self.k = k
            self.learning_data = learning_data
            self.properties = self.learning_data.iloc[:, :-1].values
            self.targets = self.learning_data.iloc[:, 4].values


    def predict(self, untagged_test_data):
        predicted_tags = []
        nearest_neighbors = []
        for x in range(len(untagged_test_data)):
            distances = []

            for y in range(len(self.learning_data)):
                distance = euclidean(self.properties[y], untagged_test_data[x])
                distances.append([distance, [self.targets[y]]])
                distances.sort()

            for i in range(self.k):
                nearest_neighbors.append(distances[i])
            nearest_neighbors.sort(key=lambda x: x[1])

            predicted_tags.append(self.getDominator(nearest_neighbors))
            nearest_neighbors.clear()

        return np.asarray(predicted_tags)

    def getDominator(self, nearest_neighbors):
        dominator_counter = 0
        dominator = nearest_neighbors[0][1]
        for i in range(self.k):
            dominator_candidate = nearest_neighbors[i][1]
            count = 0
            for j in range(self.k):
                if nearest_neighbors[j][1] == dominator_candidate:
                    count += 1
            if count > dominator_counter:
                dominator_counter = count
                dominator = dominator_candidate
        return dominator

    def score(self, untagged_test_data, test_data_targets):
        recognized = self.predict(untagged_test_data)
        correct = 0
        incorrect = 0

        for i in range(len(recognized)):

            if recognized[i] == test_data_targets[i]:
                correct += 1
            else:
                incorrect += 1

        result = correct*100/len(recognized)

        return result
