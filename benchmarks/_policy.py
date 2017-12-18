
import numpy as np

from mannequin.autograd import AutogradLayer

class SoftmaxPolicy(object):
    def __init__(self, model):
        rng = np.random.RandomState()
        eye = np.eye(model.n_outputs, dtype=np.float32)

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def sample(obs):
            outs, backprop = model.evaluate(obs)
            outs = softmax(outs)
            return rng.choice(model.n_outputs, p=outs)

        def logprob(obs, act):
            outs, backprop = model.evaluate(obs)
            assert len(outs) == len(act)
            outs = softmax(outs)
            return (
                np.log([o[a] for o, a in zip(outs, act)]),
                lambda grad: backprop(
                    np.multiply((eye[act] - outs).T, grad).T
                )
            )

        self.n_params = model.n_params
        self.get_params = model.get_params
        self.load_params = model.load_params
        self.sample = sample
        self.logprob = logprob

class GaussPolicy(AutogradLayer):
    def __init__(self, inner):
        import autograd.numpy as np

        rng = np.random.RandomState()
        logstd = np.zeros(inner.n_outputs, dtype=np.float32)

        def f(inps, logstd, *, sample):
            return -0.5 * np.sum(
                np.square((sample - inps) / np.exp(logstd)),
                axis=-1,
                keepdims=True
            ) - np.sum(logstd)

        super().__init__(inner, f=f, n_outputs=1, params=logstd)

        evaluate = self.evaluate
        del self.evaluate

        def sample(obs):
            mean, _ = inner.evaluate(obs)
            return mean + rng.randn(*mean.shape) * np.exp(logstd)

        def logprob(obs, act):
            outs, backprop = evaluate(obs, sample=act)
            return outs.reshape(-1), backprop

        self.sample = sample
        self.logprob = logprob
