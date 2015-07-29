import theano


def sigmoid(x):
    return theano.tensor.nnet.sigmoid(x)


def softmax(x):
    """
    Row-wise softmax
    """
    return theano.tensor.nnet.softmax(x)


def tanh(x):
    return theano.tensor.tanh(x)


def rectify(x):
    return 0.5 * (x + abs(x))
