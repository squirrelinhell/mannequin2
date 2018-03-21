
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

    opt = Adam(model.get_params(), horizon=10, lr=0.01)

    def sgd_step(inps, lbls):
        outs, backprop = model.evaluate(inps)
        opt.apply_gradient(backprop(lbls - outs))
        model.load_params(opt.get_value())

    model.sgd_step = sgd_step
    return model

pred = SimplePredictor(1, 1)
for _ in range(100):
    x = np.random.randn(128, 1) * 2.0
    pred.sgd_step(x, np.sin(x))

assert pred([1.0]).shape == (1,)

x = np.linspace(-5.0, 5.0, 101).reshape(-1, 1)
y = pred(x)
assert y.shape == x.shape
error = np.mean(np.abs(y - np.sin(x)))
sys.stderr.write("Mean error: %.4f\n" % error)
assert error < 0.2

assert timer() < 0.3
