
import sys
import numpy as np
from mannequin import SimplePredictor
from test_setup import timer

pred = SimplePredictor()
for _ in range(100):
    x = np.random.randn(128) * 2.0
    pred.sgd_step(x, np.sin(x))

x = np.linspace(-5.0, 5.0, 101)
error = np.mean(np.abs(pred.predict(x) - np.sin(x)))
sys.stderr.write("Mean error: %.4f\n" % error)
assert error < 0.2

assert timer() < 0.3
