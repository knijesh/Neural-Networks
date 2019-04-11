from OutputError import OutputError
import numpy as np


class SquaredError(OutputError):
    def __init__(self, key, output, error):
        self.output = output[key]
        self.error = error

    def transform(self, record):
        return 1.0 * record

    def errorDelta(self):
        transform = np.vectorize(self.transform)
        error = transform(self.error)
        return error.T