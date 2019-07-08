
import numpy as np


class Neural_Network(object):
    def __init__(self, architecture, error, indepdata, depdata):
        self.index = indepdata.index.values
        self.indepData = np.mat(np.atleast_2d(indepdata))
        self.depData = np.mat(np.atleast_2d(depdata))
        self.error = error[0]
        self.check =[]

        self.architecture = architecture
        self.architecture[0] = ["Identity", int(self.indepData.shape[1])]
        self.architecture[len(self.architecture)] = [error[1], int(self.depData.shape[1])]
        print "\n"
        print "Setting ANN with the following structure: {Layer : [Activation, Nodes]}"
        print self.architecture
        print "\n"

    def initlayer(self):
        self.layers = {}
        # Add Hidden Layers
        for x in range(1, len(self.architecture) - 1):
            layer = self.architecture[x][0]
            exec ("self.layers[x] = %s(self.architecture[x][1],False)" % layer)

        # Add Output Layers
        x = len(self.architecture) - 1
        layer = self.architecture[x][0]
        exec ("self.layers[x] = %s(self.architecture[x][1], True)" % layer)

        return self.layers











