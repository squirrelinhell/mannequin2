
import numpy as np

class Input(object):
    def __init__(self, n_inputs):
        def load_params(params):
            assert len(params) == 0

        self.n_outputs = n_inputs
        self.evaluate = lambda inps: (inps, lambda grad: [])
        self.n_params = 0
        self.load_params = load_params

class Layer(object):
    def __init__(self, inner, *, evaluate):
        self.n_outputs = inner.n_outputs
        self.evaluate = evaluate
        self.n_params = inner.n_params
        self.load_params = inner.load_params

class ParametrizedLayer(Layer):
    def __init__(self, inner, *, evaluate, params):
        def load_params(new_val):
            new_val = np.asarray(new_val, dtype=np.float32)
            assert len(new_val.shape) == 1
            params[:] = new_val[-params.size:].reshape(params.shape)
            inner.load_params(new_val[:-params.size])

        super().__init__(inner, evaluate=evaluate)
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

        super().__init__(inner, evaluate=evaluate, params=params)
        self.n_outputs = n_outputs

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

        super().__init__(inner, evaluate=evaluate, params=params)

class Multiplier(Layer):
    def __init__(self, inner, multiplier):
        multiplier = np.asarray(multiplier, dtype=np.float32)

        def evaluate(inps):
            inps, inner_backprop = inner.evaluate(inps)
            def backprop(grad):
                return inner_backprop(
                    np.multiply(grad, multiplier)
                )
            return np.multiply(inps, multiplier), backprop

        super().__init__(inner, evaluate=evaluate)

class LReLU(Layer):
    def __init__(self, inner, *, leak=0.1):
        leak = float(leak)

        def evaluate(inps):
            inps, inner_backprop = inner.evaluate(inps)
            negative = inps < 0.0
            def backprop(grad):
                grad = np.array(grad)
                grad[negative] *= leak
                return inner_backprop(grad)
            inps = np.array(inps)
            inps[negative] *= leak
            return inps, backprop

        super().__init__(inner, evaluate=evaluate)

class Tanh(Layer):
    def __init__(self, inner):
        def evaluate(inps):
            inps, inner_backprop = inner.evaluate(inps)
            tanh = np.tanh(inps)
            def backprop(grad):
                dtanh = 1.0 - np.square(tanh)
                return inner_backprop(grad * dtanh)
            return tanh, backprop

        super().__init__(inner, evaluate=evaluate)

def Affine(inner, n_outputs):
    return Multiplier(
        Bias(Linear(inner, n_outputs)),
        1.0 / np.sqrt(inner.n_outputs + 1)
    )
