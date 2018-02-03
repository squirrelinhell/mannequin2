
import numpy as np

def autograd(f):
    import autograd
    import autograd.numpy as np
    df = autograd.grad(lambda a, ka, g: np.sum(f(*a, **ka) * g))
    return lambda *a, **ka: (f(*a, **ka), lambda g: df(a, ka, g))

def add(a, b):
    ash, bsh = a.shape, b.shape
    if ash == bsh:
        return a + b, lambda g: (g, g)
    elif len(ash) > len(bsh) and ash[len(ash)-len(bsh):] == bsh:
        return a + b, lambda g: (
            g,
            np.sum(g, axis=tuple(range(len(ash)-len(bsh))))
        )
    else:
        raise ValueError("Invalid shapes: %s, %s" % (a.shape, b.shape))

def multiply(a, b):
    assert a.shape == b.shape
    return a * b, lambda g: (g * b, g * a)

def matmul(a, b):
    assert len(a.shape) in (1, 2)
    assert len(b.shape) == 2
    return np.matmul(a, b), lambda g: (
        np.matmul(g, b.T),
        np.matmul(a.T, g) if len(a.shape) >= 2 else a[:,None] * g
    )

def relu(a, *, leak=0.0):
    multiplier = np.ones(a.shape)
    multiplier[a < 0.0] = float(leak)
    return a * multiplier, lambda g: (g * multiplier,)

def tanh(a):
    a = np.tanh(a)
    return a, lambda g: (g * (1.0 - np.square(a)),)
