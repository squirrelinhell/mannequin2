
import inspect
import numpy as np
import mannequin.backprop as backprop

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
            f, n_outputs=None, params=None):
        if n_outputs is None:
            n_outputs = inner.n_outputs

        f_spec = inspect.getfullargspec(f)

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

        def evaluate(inps, **kwargs):
            inps = np.asarray(inps, dtype=np.float32)
            inps, inner_backprop = inner.evaluate(inps)
            assert inps.shape[-1] == inner.n_outputs

            if params is None:
                f_value, f_backprop = f(inps, **kwargs)
            else:
                f_value, f_backprop = f(inps, params, **kwargs)
            assert f_value.shape[-1] == n_outputs
            f_value.setflags(write=False)

            def backprop(grad):
                grad = np.asarray(grad, dtype=np.float32)
                assert grad.shape == f_value.shape
                grad = f_backprop(grad)

                inps_grad = grad[f_spec.args[0]]
                assert inps_grad.shape == inps.shape

                if params is None:
                    return inner_backprop(inps_grad)
                else:
                    params_grad = grad[f_spec.args[1]]
                    assert params_grad.shape == params.shape

                    if len(f_value.shape) >= 2:
                        params_grad /= np.prod(f_value.shape[:-1])

                    return np.concatenate((
                        inner_backprop(inps_grad),
                        params_grad.reshape(-1)
                    ), axis=0)

            return f_value, backprop

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

def normed_columns(inps, outs):
    m = np.random.randn(inps, outs).astype(np.float32)
    return m / np.sqrt(np.sum(np.square(m), axis=0))

def Linear(inner, n_outputs, *, init=normed_columns):
    if callable(init):
        params = init(inner.n_outputs, n_outputs).astype(np.float32)
    else:
        params = normed_columns(inner.n_outputs, n_outputs)
        params *= float(init)

    return Layer(inner, f=backprop.matmul,
        n_outputs=n_outputs, params=params)

def Bias(inner, *, init=np.zeros):
    params = init(inner.n_outputs).astype(np.float32)
    return Layer(inner, f=backprop.add, params=params)

def LReLU(inner, *, leak=0.1):
    return Layer(inner, f=lambda a: backprop.relu(a, leak=leak))

def Tanh(inner):
    return Layer(inner, f=backprop.tanh)

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
