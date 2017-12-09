
import numpy as np

from test_setup import timer
from mannequin import SimplePredictor, bar

pred = SimplePredictor()
for _ in range(100):
    x = np.random.randn(128) * 2.0
    pred.sgd_step(x, np.sin(x))

x = np.linspace(-5.0, 5.0, 101)
error = np.mean(np.abs(pred.predict(x) - np.sin(x)))
assert error < 0.2

assert timer() < 0.3
