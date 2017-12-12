
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

class Linear(object):
    def __init__(self, inner, n_outputs):
        n_outputs = int(n_outputs)
        n_inputs = int(inner.n_outputs)
        params = np.zeros((n_inputs, n_outputs), dtype=np.float32)

        def load_params(new_params):
            new_params = np.asarray(new_params, dtype=np.float32)
            assert len(new_params) >= params.size
            params[:] = new_params[-params.size:].reshape(params.shape)
            inner.load_params(new_params[:-params.size])

        def evaluate(inputs):
            inputs, inner_backprop = inner.evaluate(inputs)
            def backprop(grad):
                nonlocal inputs
                grad = np.asarray(grad, dtype=np.float32)
                if len(inputs.shape) <= 1:
                    inputs = np.reshape(inputs, (1,) + inputs.shape)
                if len(grad.shape) <= 1:
                    grad = np.reshape(grad, (1,) + grad.shape)
                return np.concatenate((
                    inner_backprop(np.dot(grad, params.T)),
                    np.dot(inputs.T, grad).reshape(-1) / len(grad)
                ), axis=0)
            return np.dot(inputs, params), backprop

        self.n_params = inner.n_params + params.size
        self.n_outputs = n_outputs
        self.load_params = load_params
        self.evaluate = evaluate

class Bias(object):
    def __init__(self, inner):
        params = np.zeros(int(inner.n_outputs), dtype=np.float32)

        def load_params(new_params):
            new_params = np.asarray(new_params, dtype=np.float32)
            assert len(new_params) >= params.size
            params[:] = new_params[-params.size:]
            inner.load_params(new_params[:-params.size])

        def evaluate(input_batch):
            input_batch, inner_backprop = inner.evaluate(input_batch)
            def backprop(grad):
                return np.concatenate((
                    inner_backprop(grad),
                    np.mean(grad, axis=tuple(range(0, len(grad.shape)-1)))
                ), axis=0)
            return input_batch + params, backprop

        self.n_params = inner.n_params + params.size
        self.n_outputs = params.size
        self.load_params = load_params
        self.evaluate = evaluate

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
