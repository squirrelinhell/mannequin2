#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("../..")
from mannequin import Adam
from mannequin.basicnet import Input, Affine, LReLU

def softmax(v):
    v = v.T
    v = np.exp(v - np.amax(v, axis=0))
    v /= np.sum(v, axis=0)
    return v.T

def accuracy(a, b):
    assert a.shape == b.shape
    assert len(a.shape) == 2
    return np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1))

def run():
    run_id = 0 ### 0 / 1 / 2 / 3

    data = np.load("__data.npz")
    train = data["train_x"], data["train_y"]
    test = data["test_x"], data["test_y"]

    model = Input(28 * 28)
    model = LReLU(Affine(model, 128))
    model = LReLU(Affine(model, 128))
    model = Affine(model, 10)

    rng = np.random.RandomState()
    opt = Adam(
        rng.randn(model.n_params) * 0.1,
        lr=0.04
    )

    print("# type batch run_id accuracy")

    for batch in range(1, 5001):
        model.load_params(opt.get_value())

        idx = rng.choice(len(train[0]), size=128)
        inps = train[0][idx]
        labels = train[1][idx]

        outs, backprop = model.evaluate(inps)
        grad = labels - softmax(outs) - 0.01 * outs
        opt.apply_gradient(backprop(grad))

        print("train", batch, run_id, accuracy(outs, labels))

        if batch % 100 == 0:
            model.load_params(opt.get_value())
            outs, _ = model.evaluate(test[0])

            print("test", batch, run_id, accuracy(outs, test[1]))

if __name__ == "__main__":
    run()
