
from mannequin.basicnet import Layer

class Discrete(Layer):
    def __init__(self, *, logits):
        import numpy as np

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
                np.log([[o[s]] for o, s in zip(probs, sample)]),
                lambda g: {
                    "logits": np.multiply(
                        (eye[sample] - probs).T,
                        np.reshape(g, -1)
                    ).T
                }
            )

        super().__init__(logits, f=f, n_outputs=1)
        self.sample = sample

class Gauss(Layer):
    def __init__(self, *, mean):
        import autograd.numpy as np
        from mannequin.autograd import wrap

        rng = np.random.RandomState()
        logstd = np.zeros(mean.n_outputs, dtype=np.float32)
        const = 0.5 * len(logstd) * np.log(2.0 * np.pi)

        def sample(obs):
            m = mean(obs)
            return m + rng.randn(*m.shape) * np.exp(logstd)

        def f(mean, logstd, *, sample):
            return -0.5 * np.sum(
                np.square((sample - mean) / np.exp(logstd)),
                axis=-1,
                keepdims=True
            ) - (np.sum(logstd) + const)

        super().__init__(mean, f=wrap(f), n_outputs=1, params=logstd)
        self.sample = sample
