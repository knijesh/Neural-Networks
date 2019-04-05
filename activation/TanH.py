from Activation import Activation
import math

from com.aexp.ml.ann.activation.Sigmoid import Sigmoid


class TanH(Sigmoid):

    def __init__(self):
        return

    def activation(self,record):
        ex = math.exp(record);
        exInv = math.exp(-1 * record)
        return (ex - exInv) / (ex + exInv)