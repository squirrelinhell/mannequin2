
import sys
import numpy as np
from mannequin import SimplePredictor
from test_setup import timer

pred = SimplePredictor()
for _ in range(100):
    x = np.random.randn(128) * 2.0
    pred.sgd_step(x, np.sin(x))

assert pred.predict(1.0).shape == ()

x = np.linspace(-5.0, 5.0, 101)
y = pred.predict(x)
assert y.shape == x.shape
error = np.mean(np.abs(y - np.sin(x)))
sys.stderr.write("Mean error: %.4f\n" % error)
assert error < 0.2

assert timer() < 0.31
