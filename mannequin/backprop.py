
import inspect
import decorator
import itertools
import numpy as np

def wrap(f=None, *, broadcast=False):
    broadcast = bool(broadcast)
    if f is None:
        return lambda f: wrap(f, broadcast=broadcast)

    f_spec = inspect.getfullargspec(f)

    def broadcasted_dims(orig, new):
        expand = len(new) - len(orig)
        return list(range(expand)) + [expand + i
            for i, s in enumerate(orig) if new[expand + i] != s]

    def wrapper(f, *args, **kwargs):
        assert len(f_spec.args) >= len(args)

        arg_values = {}
        arg_shapes = {}
        arg_bpp = {}

        for arg, v in itertools.chain(kwargs.items(),
                zip(f_spec.args, args)):
            if isinstance(v, tuple) and len(v) == 2 and callable(v[1]):
                arg_bpp[arg] = v[1]
                v = v[0]
            v = np.asarray(v, dtype=np.float32)
            arg_values[arg] = v
            arg_shapes[arg] = v.shape

        broadcast_shape = None
        if broadcast:
            broadcast_shape = np.broadcast(*arg_values.values()).shape
            arg_values = {k: np.broadcast_to(v, broadcast_shape)
                for k, v in arg_values.items()}

        output, f_bpp = f(**arg_values)

        output = np.asarray(output, dtype=np.float32)
        output.setflags(write=False)
        output_shape = output.shape

        def output_backprop(output_grad):
            output_grad = np.asarray(output_grad, dtype=np.float32)
            assert output_grad.shape == output_shape

            arg_grad = f_bpp(output_grad)
            ret = {}

            for arg, grad in arg_grad.items():
                grad = np.asarray(grad, dtype=np.float32)

                if broadcast_shape:
                    assert grad.shape == broadcast_shape
                    ax = broadcasted_dims(arg_shapes[arg], grad.shape)
                    grad = np.sum(grad, axis=tuple(ax))

                assert grad.shape == arg_shapes[arg]

                if arg in arg_bpp:
                    v = arg_bpp[arg](grad)
                    if v is not None and v != {}:
                        ret[arg] = v
                elif arg in arg_values:
                    ret[arg] = grad

            return ret

        return output, output_backprop

    return decorator.decorate(f, wrapper)

@wrap(broadcast=True)
def add(a, b):
    return a + b, lambda g: {"a": g, "b": g}

@wrap(broadcast=True)
def multiply(a, b):
    return a * b, lambda g: {"a": g * b, "b": g * a}

@wrap
def matmul(a, b):
    assert len(a.shape) in (1, 2)
    assert len(b.shape) == 2
    return np.matmul(a, b), lambda g: {
        "a": np.matmul(g, b.T),
        "b": np.matmul(a.T, g) if len(a.shape) == 2 else a[:,None] * g
    }

@wrap
def relu(a, *, leak=0.0):
    multiplier = np.ones(a.shape)
    multiplier[a < 0.0] = float(leak)
    return a * multiplier, lambda g: {"a": g * multiplier}

@wrap
def tanh(a):
    a = np.tanh(a)
    return a, lambda g: {"a": g * (1.0 - np.square(a))}
