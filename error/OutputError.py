import abc as abc

class OutputError(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def transform(self, record):
        """Score a single unit of the data vector. To be implemented for each activation type."""
        raise NotImplementedError()

    @abc.abstractmethod
    def errorDelta(self, data):
        """Vectorize the transform function and apply the error calculation on input data."""
        raise NotImplementedError()
