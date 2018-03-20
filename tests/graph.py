
import numpy as np
from mannequin.basicnet import *
from test_setup import timer

def PrintLayer(inner):
    return Function(inner, f=lambda a: (
        (print("Forward pass:", a), a)[1], lambda g: (g,)
    ))

def test(m, inps=[2., 3.]):
    inps = np.array(inps)
    val, bpp = m.evaluate(inps)
    print(val, bpp(np.ones(inps.shape)))

a = Bias(PrintLayer(Input(2)))
b = Tanh(a)

test(a)
test(b)
test(a + b)
test(a * a)

assert timer(print_info=False) < 0.01
