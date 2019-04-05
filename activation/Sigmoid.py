from Activation import Activation
import math
import numpy as np
import scipy
from com.aexp.ml.ann.error.OutputError import OutputError
from com.aexp.ml.ann.error.CrossEntropyError import CrossEntropyError
from com.aexp.ml.ann.error.SquaredError import SquaredError


class Sigmoid(Activation):
    def __init__(self, nodes, isOutput):
        self.nodes = nodes
        self.isOutput = isOutput

    def activation(self, x):
        return (1.0) / (1.0 + math.exp(-1 * x))

    def subtractOnes(self, array):
        ones = np.ones(array.shape)
        return np.mat(ones - array)

    def scoreData(self, data, weights):
        activation = np.vectorize(self.activation)
        net = data * weights
        score = activation(net)

        return score

    def delLayer(self, layer, weights, hidden, error, delta, function):
        deltaLayer = None
        deltaCarry = None

        thisWeights = weights[layer]
        thisHidden = hidden[layer]
        thisHidden_ = self.subtractOnes(thisHidden)

        if self.isOutput == True:
            exec ("delta = %s(layer, hidden, error).errorDelta()" % function)

        deltaLayer = np.multiply(np.multiply(thisHidden, thisHidden_), delta.T)
        deltaCarry = scipy.delete((thisWeights * deltaLayer.T), 0, axis=0)

        return deltaLayer, deltaCarry