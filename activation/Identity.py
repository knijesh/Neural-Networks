from com.aexp.ml.ann.activation.Activation import Activation
from com.aexp.ml.ann.activation.Sigmoid import Sigmoid


class Identity(Sigmoid):

    def __init__(self):
        return

    def score(self,record):
        return record