
import inspect
import autograd
import autograd.numpy as np

from mannequin.basicnet import Layer, Input, equalized_columns

class AutogradLayer(Layer):
    def __init__(self, inner, *, f, params=None, **layer_args):
        f_args_names = inspect.getfullargspec(f).kwonlyargs
        df = autograd.grad(lambda a, ka, g: np.sum(f(*a, **ka) * g))

        def evaluate(inps, **args):
            f_args = {}
            for a in f_args_names:
                if not a in args:
                    raise TypeError("evaluate() missing"
                        + " required argument: '%s'" % a)
                f_args[a] = args[a]
                del args[a]
            inps = np.asarray(inps, dtype=np.float32)
            inps, inner_backprop = inner.evaluate(inps, **args)
            assert inps.shape == (*inps.shape[:-1], inner.n_outputs)
            if params is None:
                outs = f(inps, **f_args)
            else:
                outs = f(inps, params, **f_args)
            outs = np.asarray(outs, dtype=np.float32)
            assert outs.shape == (*inps.shape[:-1], self.n_outputs)
            def backprop(grad):
                nonlocal inps
                inps = np.reshape(inps, (-1, inner.n_outputs))
                grad = np.asarray(grad, dtype=np.float32)
                grad = np.reshape(grad, (-1, self.n_outputs))
                if params is None:
                    i_grad, = df((inps,), f_args, grad)
                    return inner_backprop(i_grad)
                else:
                    i_grad, p_grad = df((inps, params), f_args, grad)
                    return np.concatenate((
                        inner_backprop(i_grad),
                        p_grad.reshape(-1) / len(grad)
                    ), axis=0)
            return outs, backprop

        super().__init__(inner, evaluate=evaluate,
            params=params, **layer_args)

def Linear(inner, n_outputs, *, init=equalized_columns):
    params = init(inner.n_outputs, n_outputs).astype(np.float32)
    multiplier = 1.0 / np.sqrt(float(inner.n_outputs))
    return AutogradLayer(inner, n_outputs=n_outputs,
        f=lambda i, w: np.dot(i, w) * multiplier, params=params)

def Bias(inner, *, init=np.zeros):
    params = init(inner.n_outputs).astype(np.float32)
    return AutogradLayer(inner, f=np.add, params=params)

def LReLU(inner, leak=0.1):
    leak = float(leak)
    return AutogradLayer(inner,
        f=lambda i: np.maximum(0.0, i) + leak * np.minimum(0.0, i))

def Tanh(inner):
    return AutogradLayer(inner, f=np.tanh)

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
