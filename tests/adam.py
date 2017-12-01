
import numpy as np

from test_setup import timer
from mannequin import Adam

opt = Adam([10.0, -5.0], lr=2.0, horizon=3)

for i in range(10):
    opt.apply_gradient(-opt.get_value())
    print(opt.get_value())

assert timer() < 0.01
