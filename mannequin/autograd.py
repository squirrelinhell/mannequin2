
import autograd
import autograd.numpy as np

from mannequin.basicnet import Layer, Input

class AutogradLayer(Layer):
    def __init__(self, inner, *, param_shape=None, f):
        params = None
        if param_shape is not None:
            param_shape = tuple(int(s) for s in param_shape)
            params = np.zeros(param_shape, dtype=np.float32)

        def load_params(new_val):
            new_val = np.asarray(new_val, dtype=np.float32)
            assert len(new_val.shape) == 1
            if params is None:
                inner.load_params(new_val)
            else:
                params[:] = new_val[-params.size:].reshape(param_shape)
                inner.load_params(new_val[:-params.size])

        df = autograd.grad(lambda args, grad: np.sum(f(*args) * grad))

        def evaluate(inps):
            inps, inner_backprop = inner.evaluate(inps)
            outs = f(inps) if params is None else f(inps, params)
            def backprop(grad):
                nonlocal inps
                inps = np.reshape(inps, (-1, inner.n_outputs))
                grad = np.asarray(grad, dtype=np.float32)
                grad = np.reshape(grad, (-1, self.n_outputs))
                if params is None:
                    i_grad, = df((inps,), grad)
                    return inner_backprop(i_grad)
                else:
                    i_grad, p_grad = df((inps, params), grad)
                    return np.concatenate((
                        inner_backprop(i_grad),
                        p_grad.reshape(-1) / len(grad)
                    ), axis=0)
            return outs, backprop

        super().__init__(inner, evaluate=evaluate)
        if param_shape is not None:
            self.n_params += params.size
        self.load_params = load_params

class Linear(AutogradLayer):
    def __init__(self, inner, n_outputs):
        super().__init__(
            inner,
            param_shape=(inner.n_outputs, n_outputs),
            f=np.dot
        )

class Bias(AutogradLayer):
    def __init__(self, inner):
        super().__init__(
            inner,
            param_shape=(inner.n_outputs,),
            f=np.add
        )

class Multiplier(AutogradLayer):
    def __init__(self, inner, multiplier):
        multiplier = np.asarray(multiplier, dtype=np.float32)

        super().__init__(
            inner,
            f=lambda i: np.multiply(i, multiplier)
        )

class LReLU(AutogradLayer):
    def __init__(self, inner, leak=0.1):
        leak = float(leak)

        super().__init__(
            inner,
            f=lambda i: np.maximum(0.0, i) + leak * np.minimum(0.0, i)
        )

class Tanh(AutogradLayer):
    def __init__(self, inner):
        super().__init__(
            inner,
            f=np.tanh
        )

def Affine(inner, n_outputs):
    return Multiplier(
        Bias(Linear(inner, n_outputs)),
        1.0 / np.sqrt(inner.n_outputs + 1)
    )
