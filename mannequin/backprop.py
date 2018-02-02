
import inspect
import decorator
import itertools
import numpy as np

def backprop(f=None, *, broadcast=False):
    broadcast = bool(broadcast)
    if f is None:
        return lambda f: backprop(f, broadcast=broadcast)

    n_args = len(inspect.getfullargspec(f).args)

    def broadcasted_dims(orig, new):
        expand = len(new) - len(orig)
        return list(range(expand)) + [expand + i
            for i, s in enumerate(orig) if new[expand + i] != s]

    def wrapper(f, *args, **kwargs):
        assert len(args) == n_args

        args = list(args)
        bpps = [None for v in args]

        for i, v in enumerate(args):
            if isinstance(v, tuple) and len(v) == 2 and callable(v[1]):
                bpps[i] = v[1]
                v = v[0]
            v = np.asarray(v, dtype=np.float32)
            args[i] = v

        shapes = [v.shape for v in args]
        if broadcast:
            broadcast_shape = np.broadcast(*args).shape
            args = [np.broadcast_to(v, broadcast_shape) for v in args]

        output, f_bpp = f(*args, **kwargs)
        output = np.asarray(output, dtype=np.float32)
        output.setflags(write=False)

        def output_backprop(output_grad):
            output_grad = np.asarray(output_grad, dtype=np.float32)
            assert output_grad.shape == output.shape

            arg_grads = f_bpp(output_grad)
            if not isinstance(arg_grads, tuple):
                arg_grads = (arg_grads,)

            assert len(arg_grads) == n_args
            ret = []

            for i, grad in enumerate(arg_grads):
                grad = np.asarray(grad, dtype=np.float32)

                if broadcast:
                    assert grad.shape == broadcast_shape
                    ax = broadcasted_dims(shapes[i], grad.shape)
                    if len(ax) >= 1:
                        grad = np.sum(grad, axis=tuple(ax))

                assert grad.shape == shapes[i]

                if bpps[i] is not None:
                    grad = bpps[i](grad)

                ret.append(grad)

            return tuple(ret)

        return output, output_backprop

    return decorator.decorate(f, wrapper)

@backprop(broadcast=True)
def add(a, b):
    return a + b, lambda g: (g, g)

@backprop(broadcast=True)
def multiply(a, b):
    return a * b, lambda g: (g * b, g * a)

@backprop
def matmul(a, b):
    assert len(a.shape) in (1, 2)
    assert len(b.shape) == 2
    return np.matmul(a, b), lambda g: (
        np.matmul(g, b.T),
        np.matmul(a.T, g) if len(a.shape) == 2 else a[:,None] * g
    )

@backprop
def relu(a, *, leak=0.0):
    multiplier = np.ones(a.shape)
    multiplier[a < 0.0] = float(leak)
    return a * multiplier, lambda g: g * multiplier,

@backprop
def tanh(a):
    a = np.tanh(a)
    return a, lambda g: g * (1.0 - np.square(a))
