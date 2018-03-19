#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("..")
from mannequin import bar
from mannequin.backprop import autograd
from mannequin.distrib import Gauss
from mannequin.basicnet import Input, Affine, Tanh, Function, Clip

def DKLUninormal(*, mean, logstd):
    @autograd
    def dkl(mean, logstd):
        import autograd.numpy as np
        return 0.5 * (
            np.sum(
                np.exp(logstd) - logstd + np.square(mean),
                axis=-1
            ) - mean.shape[-1]
        )

    return Function(mean, logstd, f=dkl, shape=())

def run():
    train_x = np.load("__mnist.npz")['train_x']

    encoder = Input(28*28)
    encoder = Tanh(Affine(encoder, 256))
    encoder = Tanh(Affine(encoder, 256))
    encoder = Gauss(
        mean=Affine(encoder, 3, init=0.1),
        logstd=Clip(Affine(encoder, 3), -6.0, 0.0)
    )

    dkl = DKLUninormal(mean=encoder.mean, logstd=encoder.logstd)

    decoder = encoder.sample
    decoder = Tanh(Affine(decoder, 256))
    decoder = Tanh(Affine(decoder, 256))
    decoder = Gauss(
        mean=Affine(decoder, 28*28, init=0.1),
        logstd=Clip(Affine(decoder, 28*28), -6.0, 0.0)
    )

    momentum = 0.0

    for i in range(10000):
        inps = train_x[np.random.choice(len(train_x), size=128)]

        logprob, backprop = decoder.logprob.evaluate(inps, sample=inps)
        grad1 = backprop(np.ones(128))

        dkl_value, backprop = dkl.evaluate(inps)
        grad2 = backprop(-np.ones(128))

        grad1[:len(grad2)] += grad2
        momentum = momentum * 0.9 + grad1 * 0.1
        momentum = np.clip(momentum, -1.0, 1.0)
        decoder.load_params(decoder.get_params() + 0.001 * momentum)

        print(
            "Logprob:", bar(np.mean(logprob), 2000.0, length=20),
            "DKL:", bar(np.mean(dkl_value), 200.0, length=20),
        )

        if i % 100 == 99:
            fig, plots = plt.subplots(3, 2)
            for inp, col in zip(inps, plots):
                for img, row in zip([inp, decoder.mean(inp)], col):
                    row.imshow(img.reshape(28,28), cmap="gray")
            fig.set_size_inches(4, 6)
            fig.savefig("step_%05d.png"%(i+1), dpi=100)
            plt.close(fig)

if __name__ == '__main__':
    run()
