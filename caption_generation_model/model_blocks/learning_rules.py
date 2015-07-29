import theano
import numpy as np
import theano.tensor as T
from itertools import izip


shared_zeros_like = lambda var: theano.shared(np.zeros_like(var.get_value()).astype(np.float32))


def get_sgd_updates(loss_expr, parameters, learning_rate):
    grads = T.grad(cost=loss_expr, wrt=parameters)
    updates = list()
    for p, g in izip(parameters, grads):
        updates.append((p, p - learning_rate * g))
    return updates


# https://github.com/lisa-lab/pylearn2/pull/136
def get_nesterov_momentum_updates(loss_expr,
                                  dense_parameters, sparse_parameters,
                                  learning_rate, momentum):
    grads = T.grad(cost=loss_expr, wrt=dense_parameters + sparse_parameters)
    dense_grads = grads[:len(dense_parameters)]
    sparse_grads = grads[-len(sparse_parameters):]
    updates = []

    for p, g in izip(sparse_parameters, sparse_grads):
        updates.append((p, p - learning_rate * g))

    for p, g in izip(dense_parameters, dense_grads):
        v = shared_zeros_like(p)
        new_v = momentum * v - learning_rate * g
        new_p = p + momentum * new_v - learning_rate * g
        updates.append((v, new_v))
        updates.append((p, new_p))
    return updates


def get_rmsprop_updates(loss_expr, dense_parameters, sparse_parameters,
                        learning_rate,  momentum, decay,
                        smoothing_coeff=np.float32(1e-4)):
    """
    RMSprop + NAG
    """
    grads = T.grad(cost=loss_expr, wrt=dense_parameters + sparse_parameters)
    dense_grads = grads[:len(dense_parameters)]
    sparse_grads = grads[-len(sparse_parameters):]
    updates = []

    for p, g in izip(sparse_parameters, sparse_grads):
        updates.append((p, p - learning_rate * g))

    for p, g in izip(dense_parameters, dense_grads):
        ms_g = shared_zeros_like(p)
        new_ms_g = decay * ms_g + (1 - decay) * T.sqr(g)
        rms_g = T.sqrt(new_ms_g + smoothing_coeff)
        g = g / rms_g
        v = shared_zeros_like(p)
        new_v = momentum * v - learning_rate * g
        new_p = p + momentum * new_v - learning_rate * g
        updates.append((ms_g, new_ms_g))
        updates.append((v, new_v))
        updates.append((p, new_p))
    return updates