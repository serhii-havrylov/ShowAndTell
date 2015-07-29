import numpy as np


class ScaleLayer(object):
    def __init__(self, name, scale_factor):
        self.name = name
        self.scale_factor = np.float32(scale_factor)

    def get_output_expr(self, input_expr):
        return self.scale_factor * input_expr

    @classmethod
    def create_copy(cls, other):
        return ScaleLayer(other.name, other.scale_factor)

    @classmethod
    def create_from_state(cls, state):
        return ScaleLayer(state['name'], state['scale_factor'])

    def get_state(self):
        return {'name': self.name,
                'scale_factor': self.scale_factor}