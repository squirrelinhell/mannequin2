
import numpy as np

from mannequin import Adam, Trajectory, RunningNormalize
from mannequin.basicnet import Input, Affine, Tanh

class SimplePredictor(object):
    def __init__(self, in_size=1, out_size=1, *,
            hid_layers=2, hid_size=64, activation=Tanh,
            classifier=False, normalize_inputs=False):
        in_size = int(in_size)
        out_size = int(out_size)
        hid_size = int(hid_size)

        # Build a simple neural network
        model = Input(in_size)
        for _ in range(hid_layers):
            model = activation(Affine(model, hid_size))
        model = Affine(model, out_size, init=0.1)

        # Use default parameter initialization
        opt = Adam(model.get_params(), horizon=10)

        if normalize_inputs:
            normalize = RunningNormalize(shape=(in_size,))

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def predict(inputs):
            inputs = np.asarray(inputs)
            if in_size == 1 and inputs.shape[-1:] != (1,):
                inputs = inputs.reshape(inputs.shape + (1,))
            if normalize_inputs:
                inputs = normalize.apply(inputs)
            outs, _ = model.evaluate(inputs)
            if classifier:
                return softmax(outs)
            elif out_size == 1:
                return outs.T[0].T
            else:
                return outs

        def sgd_step(traj, labels=None, *, lr=0.01):
            if labels is not None:
                traj = Trajectory(traj, labels)
                del labels
            assert isinstance(traj, Trajectory)
            inps = traj.o.reshape((-1, in_size))
            lbls = traj.a.reshape((-1, out_size))
            inps = normalize(inps) if normalize_inputs else inps
            outs, backprop = model.evaluate(inps)
            if classifier:
                outs = softmax(outs) + outs * 0.01
            grad = np.multiply((lbls - outs).T, traj.r).T
            opt.apply_gradient(backprop(grad), lr=lr)
            model.load_params(opt.get_value())

        self.predict = predict
        self.sgd_step = sgd_step

    def __call__(self, inputs):
        return self.predict(inputs)
