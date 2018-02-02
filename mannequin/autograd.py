
import inspect
import decorator
import autograd
import autograd.numpy as np

from mannequin.basicnet import Layer, Input, normed_columns

def wrap(f):
    f_spec = inspect.getfullargspec(f)
    df = autograd.grad(lambda args, g: np.sum(f(**args) * g))

    def wrapper(f, *args, **kwargs):
        assert len(f_spec.args) >= len(args)

        for name, v in zip(f_spec.args, args):
            assert not name in kwargs
            kwargs[name] = v

        return f(**kwargs), lambda g: df(kwargs, g)

    return decorator.decorate(f, wrapper)

def Linear(inner, n_outputs, *, init=normed_columns):
    params = init(inner.n_outputs, n_outputs).astype(np.float32)
    return Layer(inner, f=wrap(lambda a, b: np.matmul(a, b)),
        n_outputs=n_outputs, params=params)

def Bias(inner, *, init=np.zeros):
    params = init(inner.n_outputs).astype(np.float32)
    return Layer(inner, f=wrap(lambda a, b: np.add(a, b)),
        params=params)

def LReLU(inner, leak=0.1):
    leak = float(leak)
    return Layer(inner,
        f=wrap(lambda a: np.maximum(0., a) + leak * np.minimum(0., a)))

def Tanh(inner):
    return Layer(inner, f=wrap(lambda a: np.tanh(a)))

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
