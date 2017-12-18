
import numpy as np

class Input(object):
    def __init__(self, n_inputs):
        def load_params(params):
            assert len(params) == 0

        def capture_gradient(grad):
            self.last_gradient = grad
            return []

        def evaluate(inps):
            inps = np.asarray(inps, dtype=np.float32)
            if inps.shape[-1] != n_inputs:
                inps = inps.reshape((-1, n_inputs))
            return inps, capture_gradient

        self.evaluate = evaluate
        self.n_outputs = n_inputs
        self.n_params = 0
        self.get_params = lambda *, output=None: []
        self.load_params = load_params

class Layer(object):
    def __init__(self, inner, *,
            evaluate, n_outputs=None, params=None):
        if n_outputs is None:
            n_outputs = inner.n_outputs

        def get_params(*, output=None):
            if output is None:
                output = np.zeros(self.n_params, dtype=np.float32)
            assert len(output.shape) == 1
            output[-params.size:] = params.reshape(-1)
            inner.get_params(output=output[:-params.size])
            return output

        def load_params(new_val):
            new_val = np.asarray(new_val, dtype=params.dtype)
            new_val = new_val.reshape(-1)
            params[:] = new_val[-params.size:].reshape(params.shape)
            inner.load_params(new_val[:-params.size])

        self.evaluate = evaluate
        self.n_outputs = n_outputs

        if params is None:
            self.n_params = inner.n_params
            self.get_params = inner.get_params
            self.load_params = inner.load_params
        else:
            self.n_params = inner.n_params + params.size
            self.get_params = get_params
            self.load_params = load_params

    def __call__(self, inps, **kwargs):
        outs, _ = self.evaluate(inps, **kwargs)
        return outs

def equalized_columns(inps, outs):
    m = np.random.randn(inps, outs)
    return m / np.sqrt(np.mean(np.square(m), axis=0))

class Linear(Layer):
    def __init__(self, inner, n_outputs, *,
            init=equalized_columns, multiplier=None):
        params = init(inner.n_outputs, n_outputs).astype(np.float32)

        if multiplier is None:
            multiplier = 1.0 / np.sqrt(float(inner.n_outputs))
        else:
            multiplier = float(multiplier)

        def evaluate(inps, **kwargs):
            inps, inner_backprop = inner.evaluate(inps, **kwargs)
            def backprop(grad):
                nonlocal inps
                inps = np.reshape(inps, (-1, inner.n_outputs))
                grad = np.reshape(grad, (-1, n_outputs)) * multiplier
                return np.concatenate((
                    inner_backprop(np.dot(grad, params.T)),
                    np.dot(inps.T, grad).reshape(-1) / len(grad)
                ), axis=0)
            return np.dot(inps, params) * multiplier, backprop

        super().__init__(inner, evaluate=evaluate,
            n_outputs=n_outputs, params=params)

class Bias(Layer):
    def __init__(self, inner, *, init=np.zeros):
        params = init(inner.n_outputs).astype(np.float32)

        def evaluate(inps, **kwargs):
            inps, inner_backprop = inner.evaluate(inps, **kwargs)
            def backprop(grad):
                grad = np.reshape(grad, (-1, inner.n_outputs))
                return np.concatenate((
                    inner_backprop(grad),
                    np.mean(grad, axis=0)
                ), axis=0)
            return inps + params, backprop

        super().__init__(inner, evaluate=evaluate, params=params)

class Multiplier(Layer):
    def __init__(self, inner, multiplier):
        multiplier = np.asarray(multiplier, dtype=np.float32)

        def evaluate(inps, **kwargs):
            inps, inner_backprop = inner.evaluate(inps, **kwargs)
            def backprop(grad):
                return inner_backprop(np.multiply(grad, multiplier))
            return np.multiply(inps, multiplier), backprop

        super().__init__(inner, evaluate=evaluate)

class LReLU(Layer):
    def __init__(self, inner, *, leak=0.1):
        leak = float(leak)

        def evaluate(inps, **kwargs):
            inps, inner_backprop = inner.evaluate(inps, **kwargs)
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
        def evaluate(inps, **kwargs):
            inps, inner_backprop = inner.evaluate(inps, **kwargs)
            tanh = np.tanh(inps)
            def backprop(grad):
                dtanh = 1.0 - np.square(tanh)
                return inner_backprop(grad * dtanh)
            return tanh, backprop

        super().__init__(inner, evaluate=evaluate)

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
