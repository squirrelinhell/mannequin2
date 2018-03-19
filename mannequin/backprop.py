
import numpy as np

def endswith(a, b):
    if len(a) < len(b):
        return False
    return a[len(a)-len(b):] == b

def autograd(f):
    import autograd
    import autograd.numpy as np
    df = autograd.grad(lambda a, ka, g: np.sum(f(*a, **ka) * g))
    return lambda *a, **ka: (f(*a, **ka), lambda g: df(a, ka, g))

def add(a, b):
    ash, bsh = a.shape, b.shape
    if ash == bsh:
        return a + b, lambda g: (g, g)
    elif endswith(ash, bsh):
        return a + b, lambda g: (
            g,
            np.sum(g, axis=tuple(range(len(ash)-len(bsh))))
        )
    elif endswith(bsh, ash):
        return a + b, lambda g: (
            np.sum(g, axis=tuple(range(len(bsh)-len(ash)))),
            g
        )
    else:
        raise ValueError("Invalid shapes: %s, %s" % (a.shape, b.shape))

def multiply(a, b):
    ash, bsh = a.shape, b.shape
    if ash == bsh:
        return a * b, lambda g: (g * b, g * a)
    elif endswith(ash, bsh):
        return a * b, lambda g: (
            g * b,
            np.sum(g * a, axis=tuple(range(len(ash)-len(bsh))))
        )
    elif endswith(bsh, ash):
        return a * b, lambda g: (
            np.sum(g * a, axis=tuple(range(len(bsh)-len(ash)))),
            g * b
        )
    else:
        raise ValueError("Invalid shapes: %s, %s" % (a.shape, b.shape))

def clip(v, *, a, b):
   multiplier = np.ones(v.shape)
   multiplier[v < a] = 0.0
   multiplier[v > b] = 0.0
   return np.clip(v, a, b), lambda g: (g * multiplier,)

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
