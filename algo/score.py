import numpy as np

from utils.utils import Utils


class Score(object):

    def scoreData(self, weights, indepData,layers, depData = None, isError = False):
        self.utils = Utils()
        thisInput = indepData
        scores = {}
        error = None
        self.layers = layers
        scores[0] = indepData

        self.keys = weights.keys()
        for x in self.keys:
            thisWeight = weights[x]
            thisLayer  = self.layers[x]
            thisInput  = thisLayer.scoreData(self.utils.addBias(thisInput) ,thisWeight)
            scores[x] = thisInput

        if isError:
            output = scores[len(scores) - 1]
            error = np.mat(output - depData)

            for row ,column in zip(*np.where(np.isnan(error))):
                error[row ,column] = 0.0

        return scores, error