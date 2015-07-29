import theano
import theano.tensor as T


class DenseLayer(object):
    def __init__(self, name, W_init, b_init=None):
        self.name = name
        self.W = theano.shared(value=W_init(), name=name + '_W')
        if b_init:
            self.b = theano.shared(value=b_init(), name=name + '_b')

    def get_output_expr(self, input_expr):
        if hasattr(self, 'b'):
            return T.dot(input_expr, self.W) + self.b
        else:
            return T.dot(input_expr, self.W)

    def get_parameters(self):
        if hasattr(self, 'b'):
            return [self.W, self.b]
        else:
            return [self.W]

    @classmethod
    def create_copy(cls, other):
        self = cls.__new__(cls)
        self.name = other.name
        self.W = other.W
        if hasattr(other, 'b'):
            self.b = other.b
        return self

    @classmethod
    def create_from_state(cls, state):
        if state['b'] is not None:
            return DenseLayer(state['name'], lambda: state['W'], lambda: state['b'])
        return DenseLayer(state['name'], lambda: state['W'])

    def get_state(self):
        state = {}
        state['name'] = self.name
        state['W'] = self.W.get_value()
        if hasattr(self, 'b'):
            state['b'] = self.b.get_value()
        else:
            state['b'] = None
        return state