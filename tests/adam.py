
import numpy as np

from test_setup import timer
from mannequin import Adam

opt = Adam([10.0, -5.0], lr=2.0, horizon=3)
for i in range(10):
    opt.apply_gradient(-opt.get_value())
    print(opt.get_value())

print()

opt = Adam([0.0, 0.0, 0.0], horizon=1)
for i in range(10):
    opt.apply_gradient([0.1, 1.0, 10.0] - opt.get_value(), lr=7.0/(i+1))
    print(opt.get_value())

assert timer() < 0.02
