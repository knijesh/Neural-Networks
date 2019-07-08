import math

import numpy as np

from FFbackprop.backprop import Score
from utils.utils import Utils


class Train(object):

    def initParameters(self, regularization, momentum, batchsize, iteration, architecture,lrf):

        self.regularization = regularization
        self.momentum = momentum
        self.batchsize = batchsize
        self.iteration = iteration
        self.architecture = architecture
        self.lrf = lrf

        self.learningRate = {}

        for x in range(1, len(self.architecture)):

            nodes = self.architecture[x - 1][1] + 1
            rate = (1.0 - self.momentum) / math.sqrt(nodes)
            values = []

            for y in range(iteration):
                #                 factor = math.log(y + 1)
                #                 factor = math.sqrt(y )
                factor = y
                values.append(1.0 * rate / (2 ** (factor)))

            self.learningRate[x] = values
          

    def weightsupdate(self, weights, previousUpdates, iteration, observation, indepData, depData, layers):
        self.score = Score()
        self.indepData = indepData
        self.depData = depData
        self.layers = layers

        indepData = self.indepData[observation, :]

        self.utils = Utils()
        depData = self.depData[observation, :]

        cumdeltaWeights = self.utils.initZeroes(self.architecture, weights)
        modWeights = self.utils.initZeroes(self.architecture, weights)
        updates = self.utils.initZeroes(self.architecture, weights)

        for x in range(indepData.shape[0]):

            indepData_ = indepData[x]
            depData_ = depData[x]

            output_, error_ = self.score.scoreData(weights, indepData_, depData_, True)

            deltaWeights = {}

            for y in reversed(sorted(self.layers.keys())):

                layer = self.layers[y]

                if layer.isOutput == True:
                    deltaLayer, deltaCarry = layer.delLayer(y, weights, output_, error_, None, self.error)

                else:
                    deltaLayer, deltaCarry = layer.delLayer(y, weights, output_, None, deltaCarry, None)

                hidden = self.utils.addBias(output_[y - 1])
                deltaWeights[y] = hidden.T * deltaLayer

                totDeltaWeights[y] = totDeltaWeights[y] + (1.0 / len(observation)) * deltaWeights[y]

        for z in self.layers.keys():
            updates[z] = -1.0 * self.learningRate[z][iteration] * (
                    totDeltaWeights[z] + self.regularization * weights[z])
            updates[z] = updates[z] + self.momentum * previousUpdates[z]

            newWeights[z] = weights[z] + updates[z]

        return newWeights, updates

    def trainData(self, weights, index, indepData, depData, layers):
        self.utils = Utils()
        self.index = index

        for i in range(self.iteration):

            print "Iteration: " + str(i)

            updates = self.utils.initZeroes(self.architecture, weights)
            random = np.random.RandomState(1008 + i)
            remainder = len(self.index) - self.batchsize * int(len(self.index) / self.batchsize)
            odd = np.ravel(random.choice(self.index, [1, remainder], replace=False).tolist()).tolist()
            index = list(set(self.index) - set(odd))

            choice = random.choice(index, [len(self.index) / self.batchsize, self.batchsize], replace=False).tolist()
            choice.append(odd)

            choice = list(reversed(choice))

            for element in choice:
                if len(element) > 0:
                    try:
                        weights, updates = self.updateWeights(weights, updates, i, element, indepData, depData, layers)
                    except IndexError:
                        pass

        return weights
