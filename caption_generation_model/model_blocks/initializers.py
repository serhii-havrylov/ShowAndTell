import numpy as np
from numpy import random


r = random.RandomState(seed=42)


class Constant(object):
    def __init__(self, shape, val=0.0):
        self.shape = shape
        self.val = val

    def __call__(self):
        c = np.empty(self.shape, dtype=np.float32)
        c.fill(self.val)
        return c


class Normal(object):
    def __init__(self, shape, std=0.01, mean=0.0):
        self.shape = shape
        self.std = std
        self.mean = mean

    def __call__(self):
        return r.normal(self.mean, self.std, size=self.shape).astype(np.float32)


class Orthogonal(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self):
        a = r.normal(0.0, 1.0, self.shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == self.shape else v
        q = q.reshape(self.shape).astype(np.float32)
        return q