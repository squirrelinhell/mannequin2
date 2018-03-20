
import numpy as np

from mannequin import endswith
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

        self.logits = logits
        self.sample = sample
        self.logprob = Function(logits, f=logprob, shape=())
        self.n_params = self.logprob.n_params
        self.get_params = self.logprob.get_params
        self.load_params = self.logprob.load_params

class Gauss(object):
    def __init__(self, *, mean, logstd=None):
        logstd = logstd or Params(*mean.shape)
        assert logstd.shape == mean.shape

        rng = np.random.RandomState()
        const = -0.5 * np.prod(mean.shape) * np.log(2.0 * np.pi)
        squash_axes = tuple([-(i+1) for i in range(len(mean.shape))])

        def sample(mean, logstd):
            noise = rng.randn(*mean.shape) * np.exp(logstd)
            if mean.shape == logstd.shape:
                return mean + noise, lambda g: (g, g * noise)
            elif endswith(mean.shape, logstd.shape):
                return mean + noise, lambda g: (g, np.sum(g * noise,
                    axis=tuple(range(len(mean.shape)-len(logstd.shape)))))
            else:
                raise ValueError("Invalid shapes: %s, %s"
                    % (mean.shape, logstd.shape))

        @autograd
        def logprob(mean, logstd, *, sample):
            import autograd.numpy as np
            return const - np.sum(
                logstd + 0.5 *
                    np.square((sample - mean) / np.exp(logstd)),
                axis=squash_axes
            )

        self.mean = mean
        self.logstd = logstd
        self.sample = Function(mean, logstd, f=sample, shape=mean.shape)
        self.logprob = Function(mean, logstd, f=logprob, shape=())
        self.n_params = self.logprob.n_params
        self.get_params = self.logprob.get_params
        self.load_params = self.logprob.load_params
