
import numpy as np

from test_setup import timer
from mannequin.basicnet import *

m = Affine(Affine(Input(2), 2), 2)
m.load_params(np.arange(12))

value, backprop = m.evaluate([1.0, 1.0])
print(value)
print(backprop([1.0, -1.0]).reshape(-1, 6))

inps = np.eye(2)
grads = [[1.0, -1.0], [-2.0, 2.0]]
value, backprop = m.evaluate(inps)
backprop_value1 = backprop(grads).reshape(-1, 6)
backprop_value2 = np.mean([
    m.evaluate(i)[1](g).reshape(-1, 6) for i, g in zip(inps, grads)
], axis=0)
print(value)
print(backprop_value1)
assert np.allclose(backprop_value1, backprop_value2)

value, backprop = LReLU(m).evaluate([[1.0, -2.3]])
print(value)
print(backprop([[1.0, -1.0]]).reshape(-1, 6))

value, backprop = Tanh(m).evaluate([[1.0, -2.3]])
print(value)
print(backprop([[1.0, -1.0]]).reshape(-1, 6))

assert timer() < 0.01
