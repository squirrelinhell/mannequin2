
import numpy as np
from mannequin.basicnet import *
from test_setup import timer

def PrintLayer(inner):
    return Function(inner, f=lambda a: (
        (print("Layer output:\n%s\n" % a), a)[1],
        lambda g: (print("Gradient:\n%s\n" % g), g)[1:]
    ))

def Clip(inner, a, b):
    def clip(v):
       def bpp(g):
           g = np.array(g)
           g[np.logical_and(v < a, g < 0.0)] = 0.0
           g[np.logical_and(v > b, g > 0.0)] = 0.0
           return (g,)
       return np.clip(v, a, b), bpp
    return Function(inner, f=clip)

a = Input(3, 2)
a = PrintLayer(a)

a = Reshape(a, (2, 3))
a = PrintLayer(a)

a = a * Params(3)
a = PrintLayer(a)

a = Clip(a, -2.5, 2.5)
a = PrintLayer(a)

a = Reshape(a, -1)
a = PrintLayer(a)

a.load_params([1, 0, -1])

val, bpp = a.evaluate(np.arange(6).reshape((1, 1, 3, 2)))
print(bpp(np.ones(val.shape)))

assert timer(print_info=False) < 0.01
