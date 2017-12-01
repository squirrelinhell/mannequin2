
import numpy as np

from test_setup import timer
from mannequin.basicnet import *

m = Affine(Affine(Input(2), 2), 2)
m.load_params(np.arange(12))

value, backprop = m.evaluate([1.0, 1.0])
print(value)
print(backprop([1.0, -1.0]))

value, backprop = m.evaluate(np.eye(2))
print(value)
print(backprop([[1.0, -1.0], [-2.0, 2.0]]).reshape(-1, 6))

value, backprop = LReLU(m).evaluate([1.0, -2.3])
print(value)

assert timer() < 0.01
