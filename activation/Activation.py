import abc as abc


class Activation(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def scoreData(self):
        'Given the weights and input data, this function will generate the scores'
        raise NotImplementedError()

    @abc.abstractmethod
    def activation(self,x):
        'Activation Function'
        raise NotImplementedError()

    @abc.abstractmethod
    def delLayer(self):
        'This function will calculate the derivatives to update the weights in any layer'
        raise NotImplementedError()


