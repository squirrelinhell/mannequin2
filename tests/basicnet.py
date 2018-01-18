
import numpy as np
import mannequin.basicnet
import mannequin.autograd
from test_setup import timer

def test_module(net):
    m = net.Affine(net.Affine(net.Input(2), 2), 2)
    m.load_params(np.arange(12))

    values = []
    def check(m, x, g):
        v, backprop = m.evaluate(x)
        return [v, backprop(g).reshape(-1, 6)]

    values += check(m, [1.0, 1.0], [1.0, -1.0])
    values += check(m, np.eye(2), [[1.0, -1.0], [-2.0, 2.0]])

    alt_value = np.mean([
        m.evaluate([1.0, 0.0])[1]([1.0, -1.0]).reshape(-1, 6),
        m.evaluate([0.0, 1.0])[1]([-2.0, 2.0]).reshape(-1, 6),
    ], axis=0)
    assert np.allclose(alt_value, values[-1])

    v = [-18.4, 2.0]
    values += [m(v)]
    values += check(net.LReLU(m), v, [1.0, -1.0])
    values += [m(v)]
    values += check(net.Tanh(m), v, [1.0, -1.0])

    return values

vs = test_module(mannequin.basicnet)
for v in vs:
    print(v)

assert timer(print_info=False) < 0.01

assert all(np.allclose(a, b) for a, b in zip(
    vs, test_module(mannequin.autograd)
))

assert timer(print_info=False) < 0.05
