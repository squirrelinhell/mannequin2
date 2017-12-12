#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow.examples.tutorials.mnist as tf_mnist

def run():
    sys.stderr.write("Downloading data...\n")

    data = tf_mnist.input_data.read_data_sets(
        "/tmp/mnist-download",
        validation_size=0,
        one_hot=True
    )

    sys.stderr.write("Saving data...\n")

    np.savez_compressed(
        "__data.npz",
        train_x=data.train.images,
        train_y=data.train.labels,
        test_x=data.test.images,
        test_y=data.test.labels
    )

if __name__ == "__main__":
    run()
