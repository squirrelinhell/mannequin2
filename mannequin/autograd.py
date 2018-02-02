
import inspect
import decorator
import autograd
import autograd.numpy as np

import mannequin.backprop
from mannequin.basicnet import Input, Params, Layer, normed_columns

def backprop(f):
    n_args = len(inspect.getfullargspec(f).args)
    df = autograd.grad(lambda a, ka, g: np.sum(f(*a, **ka) * g))

    def wrapper(f, *args, **kwargs):
        assert len(args) == n_args
        return f(*args, **kwargs), lambda g: df(args, kwargs, g)

    return mannequin.backprop.backprop(decorator.decorate(f, wrapper))

def Linear(inner, n_outputs, *, init=normed_columns):
    return Layer(inner, Params(inner.n_outputs, n_outputs, init=init),
        f=backprop(lambda a, b: np.matmul(a, b)), output_shape=(n_outputs,))

def Bias(inner, *, init=np.zeros):
    return Layer(inner, Params(*inner.output_shape, init=init),
        f=backprop(lambda a, b: np.add(a, b)))

def LReLU(inner, leak=0.1):
    leak = float(leak)
    return Layer(inner,
        f=backprop(lambda a: np.maximum(0., a) + leak * np.minimum(0., a)))

def Tanh(inner):
    return Layer(inner, f=backprop(lambda a: np.tanh(a)))

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
