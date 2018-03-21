
import sys
import numpy as np
from mannequin import Adam
from mannequin.basicnet import Input, Affine, Tanh
from test_setup import timer

def SimplePredictor(in_size, out_size):
    model = Input(in_size)
    for _ in range(2):
        model = Tanh(Affine(model, 64))
    model = Affine(model, out_size, init=0.1)

    opt = Adam(model.get_params(), horizon=10, lr=0.015)

    def sgd_step(inps, lbls):
        outs, backprop = model.evaluate(inps)
        opt.apply_gradient(backprop(lbls - outs))
        model.load_params(opt.get_value())

    model.sgd_step = sgd_step
    return model

def func(x, y):
    return [np.exp(x), x * y]

pred = SimplePredictor(2, 2)
for _ in range(500):
    xy = np.random.randn(256).reshape(-1, 2)
    pred.sgd_step(xy, [func(x, y) for x, y in xy])

assert pred(np.eye(2)).shape == (2, 2)

errors = []
for x in np.linspace(-2.0, 2.0, 21):
    for y in np.linspace(-2.0, 2.0, 21):
        p = pred([x, y])
        assert p.shape == (2,)
        errors.append(np.abs(p - func(x, y)))
errors = np.mean(errors, axis=0)
sys.stderr.write("Mean errors: %.4f %.4f\n" % tuple(errors))
assert (errors < 0.2).all()

assert timer() < 2.0
