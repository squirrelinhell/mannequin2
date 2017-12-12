
import numpy as np

class Input(object):
    def __init__(self, n_inputs):
        def load_params(params):
            assert len(params) == 0

        def evaluate(input_batch):
            input_batch = np.asarray(input_batch, dtype=np.float32)
            return input_batch, lambda output_gradients: []

        self.n_params = 0
        self.n_outputs = n_inputs
        self.load_params = load_params
        self.evaluate = evaluate

class ParametrizedLayer(object):
    def __init__(self, inner, *, params):
        def load_params(new_val):
            new_val = np.asarray(new_val, dtype=np.float32)
            assert len(new_val.shape) == 1
            params[:] = new_val[-params.size:].reshape(params.shape)
            inner.load_params(new_val[:-params.size])

        self.n_params = inner.n_params + params.size
        self.load_params = load_params

class Linear(ParametrizedLayer):
    def __init__(self, inner, n_outputs):
        n_inputs = int(inner.n_outputs)
        params = np.zeros((n_inputs, n_outputs), dtype=np.float32)

        def evaluate(inps):
            inps, inner_backprop = inner.evaluate(inps)
            def backprop(grad):
                nonlocal inps
                inps = np.reshape(inps, (-1, n_inputs))
                grad = np.asarray(grad, dtype=np.float32)
                grad = np.reshape(grad, (-1, n_outputs))
                return np.concatenate((
                    inner_backprop(np.dot(grad, params.T)),
                    np.dot(inps.T, grad).reshape(-1) / len(grad)
                ), axis=0)
            return np.dot(inps, params), backprop

        super().__init__(inner, params=params)
        self.n_outputs = n_outputs
        self.evaluate = evaluate

class Bias(ParametrizedLayer):
    def __init__(self, inner):
        n_outputs = int(inner.n_outputs)
        params = np.zeros(n_outputs, dtype=np.float32)

        def evaluate(inps):
            inps, inner_backprop = inner.evaluate(inps)
            def backprop(grad):
                grad = np.reshape(grad, (-1, n_outputs))
                return np.concatenate((
                    inner_backprop(grad),
                    np.mean(grad, axis=0)
                ), axis=0)
            return inps + params, backprop

        super().__init__(inner, params=params)
        self.evaluate = evaluate
        self.n_outputs = n_outputs

class Multiplier(object):
    def __init__(self, inner, multiplier):
        multiplier = np.asarray(multiplier, dtype=np.float32)

        def evaluate(input_batch):
            input_batch, inner_backprop = inner.evaluate(input_batch)
            def backprop(output_gradients):
                return inner_backprop(
                    np.multiply(output_gradients, multiplier)
                )
            return np.multiply(input_batch, multiplier), backprop

        self.n_params = inner.n_params
        self.n_outputs = inner.n_outputs
        self.load_params = inner.load_params
        self.evaluate = evaluate

class LReLU(object):
    def __init__(self, inner, leak=0.1):
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

class Tanh(object):
    def __init__(self, inner):
        def evaluate(input_batch):
            input_batch, inner_backprop = inner.evaluate(input_batch)
            tanh = np.tanh(input_batch)
            def backprop(output_gradients):
                dtanh = 1.0 - np.square(tanh)
                return inner_backprop(output_gradients * dtanh)
            return tanh, backprop

        self.n_params = inner.n_params
        self.n_outputs = inner.n_outputs
        self.load_params = inner.load_params
        self.evaluate = evaluate

def Affine(inner, n_outputs):
    return Multiplier(
        Bias(Linear(inner, n_outputs)),
        1.0 / np.sqrt(inner.n_outputs + 1)
    )
