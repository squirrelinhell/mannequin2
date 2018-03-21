#!/usr/bin/env python3

import sys
import numpy as np

sys.path.append("..")
from mannequin import Adam, bar
from mannequin.basicnet import Input, Affine, LReLU

def softmax(v):
    v = v.T
    v = np.exp(v - np.amax(v, axis=0))
    v /= np.sum(v, axis=0)
    return v.T

def accuracy(a, b):
    assert a.shape == b.shape
    return np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1))

def run():
    data = np.load("__mnist.npz")
    data = {k: data[k] for k in data}

    model = Input(28, 28)
    for _ in range(2):
        model = LReLU(Affine(model, 128))
    model = Affine(model, 10, init=0.1)

    opt = Adam(model.get_params(), horizon=10, lr=0.001)

    def sgd_step(inps, lbls):
        outs, backprop = model.evaluate(inps)
        grad = lbls - softmax(outs) - outs * 0.01
        opt.apply_gradient(backprop(grad))
        model.load_params(opt.get_value())

    for epoch in range(10):
        for _ in range(len(data["train_x"]) // 128):
            idx = np.random.randint(len(data["train_x"]), size=128)
            sgd_step(
                data["train_x"][idx],
                data["train_y"][idx]
            )

        idx = np.random.randint(len(data["test_x"]), size=4096)
        print(bar(100.0 * accuracy(
            model(data["test_x"][idx]),
            data["test_y"][idx]
        )))

if __name__ == '__main__':
    run()
