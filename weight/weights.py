import numpy as np
import math


class Weights(object):
    def initWeights(self, weights, isStatus=True, weight1=None, weight2=None):

        if isStatus == True:

            random = np.random.RandomState(1008)

            for x in range(1, len(self.architecture)):
                start = self.architecture[x - 1][1] + 1
                end = self.architecture[x][1]
                weights[x] = np.mat((0.1 / math.sqrt(start) * (2 * random.rand(start, end) - 1)))
        else:
            weights[1] = weight1
            weights[2] = weight2

        self.weights = weights

        return self.weights
