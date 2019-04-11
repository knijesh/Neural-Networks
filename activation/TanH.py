import math

from .Sigmoid import Sigmoid


class TanH(Sigmoid):

    def __init__(self):
        super(TanH, self).__init__()
        return

    def activation(self,record):
        ex = math.exp(record);
        exInv = math.exp(-1 * record)
        return (ex - exInv) / (ex + exInv)