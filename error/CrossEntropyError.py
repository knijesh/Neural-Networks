from OutputError import *
import numpy as np

class CrossEntropyError(OutputError):
    def __init__(self, key, output, error):
        self.output = output[key]
        self.error = error

    def transform(self, record):
        return 1.0 / (record * (1 - record))

    def errorDelta(self):
        transform = np.vectorize(self.transform)
        output = transform(self.output)
        return np.multiply(self.error, output).T