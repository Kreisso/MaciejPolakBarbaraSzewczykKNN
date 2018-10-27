from scipy.spatial.distance import pdist, squareform, distance
from scipy.spatial.distance import euclidean
import scipy as sp

class KNN:
    def __init__(self, k, list):
        self.k = k;
        self.list = list;

        self.properties = self.list.iloc[:, :-1].values
        self.targets = self.list.iloc[:, 4].values
        self.vectors = pdist(self.properties, 'euclidean')
        print(squareform(self.vectors))


    def predict(self, unTagsList):
        properties = unTagsList.list.iloc[:, :-1]

        for x in properties:
            for y in self.targets:
                dst = []
                dst.append(distance.euclidean(x, y))

        return test

    def score(self, objectList, tagsList):

        return recognize