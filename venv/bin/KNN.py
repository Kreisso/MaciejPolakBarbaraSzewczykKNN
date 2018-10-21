from scipy.spatial.distance import pdist
from scipy.spatial.distance import euclidean
import scipy as sp

class KNN:
    def __init__(self, k, list):
        self.k = k;
        self.list = list;

        self.x = self.list.iloc[:, :-1].values
        self.y = self.list.iloc[:, 4].values
        self.vectors = pdist(self.x, 'euclidean')


    def predict(self, unTagsList):

        # trainingSet = self.list;
        # testInstance = unTagsList;
        # k = self.k
        # distances = []
        # length = len(testInstance) - 1
        # for x in range(len(trainingSet)):
        #     dist = euclidean(testInstance, trainingSet[x], length)
        #     distances.append((trainingSet[x], dist))
        # distances.sort(key=operator.itemgetter(1))
        # neighbors = []
        # for x in range(k):
        #     neighbors.append(distances[x][0])
        # return neighbors




    def score(self, objectList, tagsList):

        return recognize