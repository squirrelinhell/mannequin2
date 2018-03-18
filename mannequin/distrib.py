
import numpy as np

from mannequin.basicnet import Params, Function
from mannequin.backprop import autograd

class Discrete(object):
    def __init__(self, *, logits):
        assert len(logits.shape) == 1
        n = logits.shape[0]
        rng = np.random.RandomState()
        eye = np.eye(n, dtype=np.float32)

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def sample(inps):
            return rng.choice(n, p=softmax(logits(inps)))

        def logprob(logits, *, sample):
            sample = np.asarray(sample, dtype=np.int32).reshape(-1)
            logits = logits.reshape(len(sample), n)
            probs = softmax(logits)
            return (
                np.log([o[s] for o, s in zip(probs, sample)]),
                lambda g: ((g.T * (eye[sample] - probs).T).T,)
            )

        self.sample = sample
        self.logprob = Function(logits, f=logprob, shape=())
        self.n_params = self.logprob.n_params
        self.get_params = self.logprob.get_params
        self.load_params = self.logprob.load_params

class Gauss(object):
    def __init__(self, *, mean, logstd=None):
        assert len(mean.shape) == 1
        logstd = logstd or Params(*mean.shape)
        assert logstd.shape == mean.shape

        rng = np.random.RandomState()
        const = -0.5 * mean.shape[0] * np.log(2.0 * np.pi)

        def sample(mean, logstd):
            noise = rng.randn(*mean.shape) * np.exp(logstd)
            if mean.shape == logstd.shape:
                return mean + noise, lambda g: (g, g * noise)
            elif mean.shape[1:] == logstd.shape:
                return (mean + noise,
                    lambda g: (g, np.sum(g * noise, axis=0)))
            else:
                raise ValueError("Invalid shapes: %s, %s"
                    % (mean.shape, logstd.shape))

        @autograd
        def logprob(mean, logstd, *, sample):
            import autograd.numpy as np
            return const - np.sum(
                logstd + 0.5 *
                    np.square((sample - mean) / np.exp(logstd)),
                axis=-1
            )

        self.sample = Function(mean, logstd, f=sample, shape=())
        self.logprob = Function(mean, logstd, f=logprob, shape=())
        self.n_params = self.logprob.n_params
        self.get_params = self.logprob.get_params
        self.load_params = self.logprob.load_params
