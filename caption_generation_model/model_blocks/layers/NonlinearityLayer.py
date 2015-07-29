from caption_generation_model.model_blocks import nonlinearities


class NonlinearityLayer(object):
    def __init__(self, name, nonlinearity):
        self.name = name
        self.nonlinearity = nonlinearity

    def get_output_expr(self, input_expr):
        return self.nonlinearity(input_expr)

    def get_parameters(self):
        return []

    @classmethod
    def create_copy(cls, other):
        return NonlinearityLayer(other.name, other.nonlinearity)

    @classmethod
    def create_from_state(cls, state):
        nonlinearity = getattr(nonlinearities, state['nonlinearity'])
        return NonlinearityLayer(state['name'], nonlinearity)

    def get_state(self):
        return {'name': self.name,
                'nonlinearity': self.nonlinearity.__name__}