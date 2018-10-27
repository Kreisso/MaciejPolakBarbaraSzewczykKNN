from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import euclidean
import numpy as np
import scipy as sp

class KNN:
    def __init__(self, k, list):
        self.k = k;
        self.list = list;
        self.properties = self.list.iloc[:, :-1].values
        self.targets = self.list.iloc[:, 4].values

    def predict(self, unTagsList):
        newTags = []
        nearestNeighbors = []
        for x in range(len(unTagsList)):
            distances = []

            for y in range(len(self.list)):
                dist = euclidean(self.properties[y], unTagsList[x] )
                distances.append([dist, [self.targets[y]]])
                distances.sort()

            for i in range(self.k):
                nearestNeighbors.append(distances[i])
            nearestNeighbors.sort(key=lambda x: x[1])
            maxL = 0
            maxW = nearestNeighbors[0][1]
            for i in range(self.k):
                w = nearestNeighbors[i][1]
                l = 0
                for j in range(self.k):
                    if nearestNeighbors[j][1] == w:
                        l += 1
                if l > maxL:
                    maxL = l
                    maxW = w

            newTags.append(maxW)
            nearestNeighbors.clear()

        return  np.asarray(newTags)

    def score(self, objectList, tagsList):
        correct = 0
        uncorrect = 0

        for i in range(len(objectList)):

            if objectList[i] == tagsList[i]:
                correct += 1
            else:
                uncorrect +=1

        recognized = correct*100/len(objectList )

        return recognized


