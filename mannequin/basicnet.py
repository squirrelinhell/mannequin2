
class Input(object):
    def __init__(self, n_inputs):
        import numpy as np

        def load_params(params):
            assert len(params) == 0

        def evaluate(input_batch):
            input_batch = np.asarray(input_batch, dtype=np.float32)
            return input_batch, lambda output_gradients: []

        self.n_params = 0
        self.n_outputs = n_inputs
        self.load_params = load_params
        self.evaluate = evaluate

class RawAffine(object):
    def __init__(self, inner, n_outputs):
        import numpy as np

        n_outputs = int(n_outputs)
        n_inputs = int(inner.n_outputs)
        weights = np.zeros((n_inputs, n_outputs), dtype=np.float32)
        bias = np.zeros(n_outputs, dtype=np.float32)
        n_params = int(weights.size + bias.size)

        def load_params(params):
            params = np.asarray(params, dtype=np.float32)
            assert len(params) >= n_params
            weights[:] = params[0:weights.size].reshape(weights.shape)
            bias[:] = params[weights.size:n_params]
            inner.load_params(params[n_params:])

        def evaluate(input_batch):
            input_batch, inner_backprop = inner.evaluate(input_batch)
            def backprop(output_gradients):
                output_gradients = np.asarray(output_gradients,
                    dtype=np.float32)
                return np.concatenate((
                    np.dot(input_batch.T, output_gradients).reshape(-1)
                        / len(output_gradients),
                    np.mean(output_gradients, axis=0)
                        if len(output_gradients.shape) == 2
                        else output_gradients,
                    inner_backprop(np.dot(output_gradients, weights.T))
                ), axis=0)
            return np.dot(input_batch, weights) + bias, backprop

        self.n_params = inner.n_params + n_params
        self.n_outputs = n_outputs
        self.load_params = load_params
        self.evaluate = evaluate

class Multiplier(object):
    def __init__(self, inner, multiplier):
        import numpy as np

        value = np.asarray(multiplier, dtype=np.float32)

        def evaluate(input_batch):
            input_batch, inner_backprop = inner.evaluate(input_batch)
            def backprop(output_gradients):
                return inner_backprop(output_gradients) * multiplier
            return input_batch * multiplier, backprop

        self.n_params = inner.n_params
        self.n_outputs = inner.n_outputs
        self.load_params = inner.load_params
        self.evaluate = evaluate

class LReLU(object):
    def __init__(self, inner, leak=0.1):
        import numpy as np

        leak = float(leak)

        def evaluate(input_batch):
            input_batch, inner_backprop = inner.evaluate(input_batch)
            negative = input_batch < 0.0
            def backprop(output_gradients):
                output_gradients = np.array(output_gradients)
                output_gradients[negative] *= leak
                return inner_backprop(output_gradients)
            input_batch = np.array(input_batch)
            input_batch[negative] *= leak
            return input_batch, backprop

        self.n_params = inner.n_params
        self.n_outputs = inner.n_outputs
        self.load_params = inner.load_params
        self.evaluate = evaluate

def Affine(inner, n_outputs):
    import numpy as np

    return Multiplier(
        RawAffine(inner, n_outputs),
        1.0 / np.sqrt(inner.n_outputs + 1)
    )
