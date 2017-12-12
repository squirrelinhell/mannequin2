
import numpy as np

from mannequin import Adam, Trajectory
from mannequin.basicnet import Input, Affine, Tanh

class SimplePredictor(object):
    def __init__(self, in_size=1, out_size=1, *,
            hid_layers=2, hid_size=64, classifier=False):
        # Build a simple neural network
        model = Input(int(in_size))
        for _ in range(hid_layers):
            model = Tanh(Affine(model, int(hid_size)))
        model = Affine(model, int(out_size))

        # Initialize parameters
        opt = Adam(np.random.randn(model.n_params) * 0.1)
        model.load_params(opt.get_value())

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def predict(inputs):
            inputs = np.asarray(inputs)
            if int(in_size) == 1 and inputs.shape[-1] != 1:
                inputs = inputs.reshape((-1, 1))
            outs, _ = model.evaluate(inputs)
            if classifier:
                return softmax(outs)
            elif int(out_size) == 1:
                return outs.T[0].T
            else:
                return outs

        def sgd_step(inputs, labels=None, *, lr=0.05):
            if labels is not None:
                inputs = Trajectory(inputs, labels)
            assert isinstance(inputs, Trajectory)
            outs, backprop = model.evaluate(inputs.o)
            if classifier:
                outs = softmax(outs) + outs * 0.01
            grad = np.multiply((inputs.a - outs).T, inputs.r).T
            opt.apply_gradient(backprop(grad), lr=lr)
            model.load_params(opt.get_value())

        self.predict = predict
        self.sgd_step = sgd_step
