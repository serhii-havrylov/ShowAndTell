import numpy as np
import theano.tensor as T


class DropoutLayer(object):
    def __init__(self, name, dropout_prob):
        self.name = name
        self.keep_prob = np.float32(1. - dropout_prob)

    def get_output_expr(self, input_expr):
        r_stream = T.shared_randomstreams.RandomStreams(1984)
        mask = r_stream.binomial(size=input_expr.shape, p=self.keep_prob, dtype='float32')
        return mask * input_expr

    def get_parameters(self):
        return []

    @classmethod
    def create_copy(cls, other):
        return DropoutLayer(other.name, 1.0 - other.keep_prob)

    @classmethod
    def create_from_state(cls, state):
        return DropoutLayer(state['name'], state['dropout_prob'])

    def get_state(self):
        return {'name': self.name,
                'dropout_prob': 1.0 - self.keep_prob}