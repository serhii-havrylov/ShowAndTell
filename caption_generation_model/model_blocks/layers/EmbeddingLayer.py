import theano


class EmbeddingLayer(object):
    def __init__(self, name, embedding_init):
        self.name = name
        self.embedding_matrix = theano.shared(value=embedding_init(),
                                              name=name + '_W')

    def get_output_expr(self, input_expr):
        return self.embedding_matrix[input_expr]

    def get_parameters(self):
        return [self.embedding_matrix]

    @classmethod
    def create_copy(cls, other):
        self = cls.__new__(cls)
        self.name = other.name
        self.embedding_matrix = other.embedding_matrix
        return self

    @classmethod
    def create_from_state(cls, state):
        return EmbeddingLayer(state['name'], lambda: state['embedding_matrix'])

    def get_state(self):
        return {'name': self.name,
                'embedding_matrix': self.embedding_matrix.get_value()}