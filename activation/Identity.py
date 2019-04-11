from .Sigmoid import Sigmoid


class Identity(Sigmoid):

    def __init__(self):
        super(Identity, self).__init__()
        return

    def score(self,record):
        return record