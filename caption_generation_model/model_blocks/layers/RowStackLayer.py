import theano.tensor as T


class RowStackLayer(object):
    def __init__(self, name):
        self.name = name

    def get_output_expr(self, vector, matrix):
        vector = vector.dimshuffle('x', 0)
        return T.concatenate((vector, matrix), axis=0)

    def get_parameters(self):
        return []

    @classmethod
    def create_copy(cls, other):
        return RowStackLayer(other.name)

    @classmethod
    def create_from_state(cls, state):
        return RowStackLayer(state['name'])

    def get_state(self):
        return {'name': self.name}