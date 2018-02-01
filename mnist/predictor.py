#!/usr/bin/env python3

import sys
import numpy as np

sys.path.append("..")
from mannequin import SimplePredictor, bar
from mannequin.basicnet import LReLU

def accuracy(a, b):
    assert a.shape == b.shape
    return np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1))

def run():
    data = np.load("__mnist.npz")
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    pred = SimplePredictor(28*28, 10, hid_size=128,
        classifier=True, activation=LReLU)

    for epoch in range(10):
        for _ in range(len(train_x) // 128):
            idx = np.random.randint(len(train_x), size=128)
            pred.sgd_step(train_x[idx], train_y[idx], lr=0.001)

        idx = np.random.randint(len(test_x), size=4096)
        print(bar(100.0 * accuracy(pred(test_x[idx]), test_y[idx])))

if __name__ == '__main__':
    run()
