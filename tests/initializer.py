
import sys
import numpy as np
from mannequin.basicnet import *
import mannequin.autograd
from test_setup import timer

def sample_values(Model, *, dims):
    values = []
    for _ in range(1000):
        a, b = dims()
        model = Model(Input(a), b)
        v1 = np.random.randn(a)
        v2, _ = model.evaluate(v1)
        values.append(v2.reshape(-1))
    return np.concatenate(values)

def check_std(Model, ci=(0.97, 1.03), **args):
    values = sample_values(Model, **args)
    std = np.std(values)
    sys.stderr.write("Stddev: %.2f\n" % std)
    assert (std > ci[0]) and (std < ci[1])

rand_dims = lambda: np.random.randint(16, size=2) + 5

check_std(Linear, dims=lambda: (5, 20))
check_std(Linear, dims=lambda: (20, 5))
check_std(Linear, dims=rand_dims)
check_std(Affine, dims=rand_dims)
check_std(mannequin.autograd.Affine, dims=lambda: (5, 20))
check_std(lambda *p: Linear(*p, init=10.0), dims=rand_dims, ci=(9.7, 10.3))

assert timer() < 1.0
