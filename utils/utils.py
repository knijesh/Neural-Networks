import numpy as np

class Utils(object):

    def initZeroes(self,architecture,weights):
        self.weights = weights
        self.architecture = architecture
        zeros = {}
        for x in range(1, len(self.architecture)):
            zeros[x] = np.mat(np.zeros(self.weights[x].shape))

        return zeros

    @staticmethod
    def addBias(array):
        n = array.shape[0]
        bias = np.ones((n, 1), dtype=np.float64)
        array = np.append(bias, array, axis=1)
        return array
