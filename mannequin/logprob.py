
import numpy as np

from mannequin.basicnet import Layer
from mannequin.autograd import AutogradLayer

class Discrete(Layer):
    def __init__(self, *, logits):
        rng = np.random.RandomState()
        eye = np.eye(logits.n_outputs, dtype=np.float32)

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def sample(inps):
            outs, backprop = logits.evaluate(inps)
            outs = softmax(outs)
            return rng.choice(logits.n_outputs, p=outs)

        def logprob(inps, *, sample, **kwargs):
            sample = np.asarray(sample, dtype=np.int32).reshape(-1)
            outs, backprop = logits.evaluate(inps, **kwargs)
            assert len(outs) == len(sample)
            outs = softmax(outs)
            return (
                np.log([[o[s]] for o, s in zip(outs, sample)]),
                lambda grad: backprop(np.multiply(
                    (eye[sample] - outs).T,
                    np.reshape(grad, -1)
                ).T)
            )

        super().__init__(logits, evaluate=logprob, n_outputs=1)
        self.sample = sample

class Gauss(AutogradLayer):
    def __init__(self, *, mean):
        import autograd.numpy as np

        rng = np.random.RandomState()
        logstd = np.zeros(mean.n_outputs, dtype=np.float32)
        const = 0.5 * len(logstd) * np.log(2.0 * np.pi)

        def sample(obs):
            m, _ = mean.evaluate(obs)
            return m + rng.randn(*m.shape) * np.exp(logstd)

        def logprob(mean, logstd, *, sample):
            return -0.5 * np.sum(
                np.square((sample - mean) / np.exp(logstd)),
                axis=-1,
                keepdims=True
            ) - (np.sum(logstd) + const)

        super().__init__(mean, f=logprob, n_outputs=1, params=logstd)
        self.sample = sample
