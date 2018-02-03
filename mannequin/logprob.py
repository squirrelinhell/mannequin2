
import numpy as np

from mannequin.basicnet import Params, Function
from mannequin.backprop import autograd

class Discrete(Function):
    def __init__(self, *, logits):
        n = logits.n_outputs
        rng = np.random.RandomState()
        eye = np.eye(n, dtype=np.float32)

        def softmax(v):
            v = v.T
            v = np.exp(v - np.amax(v, axis=0))
            v /= np.sum(v, axis=0)
            return v.T

        def sample(inps):
            return rng.choice(n, p=softmax(logits(inps)))

        def f(logits, *, sample):
            sample = np.asarray(sample, dtype=np.int32).reshape(-1)
            logits = logits.reshape(len(sample), n)
            probs = softmax(logits)
            return (
                np.log([o[s] for o, s in zip(probs, sample)]),
                lambda g: ((g.T * (eye[sample] - probs).T).T,)
            )

        super().__init__(logits, f=f, shape=())
        self.sample = sample

class Gauss(Function):
    def __init__(self, *, mean, logstd=None):
        import autograd.numpy as np

        logstd = logstd or Params(mean.n_outputs)
        assert logstd.shape == (mean.n_outputs,)

        rng = np.random.RandomState()
        const = -0.5 * mean.n_outputs * np.log(2.0 * np.pi)

        def sample(obs):
            m, l = mean(obs), logstd(obs)
            return m + rng.randn(*m.shape) * np.exp(l)

        @autograd
        def f(mean, logstd, *, sample):
            return const - np.sum(
                logstd + 0.5 *
                    np.square((sample - mean) / np.exp(logstd)),
                axis=-1
            )

        super().__init__(mean, logstd, f=f, shape=())
        self.sample = sample
