
import numpy as np
import mannequin.backprop as backprop

def endswith(a, b):
    if len(a) < len(b):
        return False
    return a[len(a)-len(b):] == b

class Input(object):
    def __init__(self, *shape):
        shape = tuple(max(1, int(s)) for s in shape)
        assert len(shape) >= 1

        def backprop(grad, output=None):
            assert endswith(grad.shape, shape)
            grad.setflags(write=False)
            self.last_gradient = grad[:]
            return get_params(output=output)

        def evaluate(inps):
            inps = np.asarray(inps, dtype=np.float32)
            assert endswith(inps.shape, shape)
            return inps, backprop

        def get_params(*, output=None):
            if output is None:
                return []
            else:
                assert len(output) == 0
                return output

        def load_params(params):
            assert len(params) == 0

        self.evaluate = evaluate
        self.output_shape = shape
        self.n_params = 0
        self.get_params = get_params
        self.load_params = load_params

        if len(self.output_shape) == 1:
            self.n_outputs = self.output_shape[0]

    def __call__(self, inps):
        outs, _ = self.evaluate(inps)
        return outs

class Params(object):
    def __init__(self, *shape, init=np.zeros):
        shape = tuple(max(1, int(s)) for s in shape)
        assert len(shape) >= 1

        value = np.array(init(*shape), dtype=np.float32).reshape(shape)
        value.setflags(write=False)

        def backprop(grad, output=None):
            if output is None:
                return grad.reshape(value.size)
            else:
                output[:] = grad.reshape(value.size)
                return output

        def evaluate(inps):
            return value[:], backprop

        def get_params(*, output=None):
            if output is None:
                return value.reshape(-1)
            else:
                output[:] = value.reshape(-1)
                return output

        def load_params(new_value):
            value.setflags(write=True)
            value[:] = new_value.reshape(shape)
            value.setflags(write=False)

        self.evaluate = evaluate
        self.output_shape = shape
        self.n_params = value.size
        self.get_params = get_params
        self.load_params = load_params

        if len(self.output_shape) == 1:
            self.n_outputs = self.output_shape[0]

    def __call__(self, inps):
        outs, _ = self.evaluate(inps)
        return outs

class Layer(object):
    def __init__(self, *args, f, output_shape=None, params=None):
        assert len(args) >= 1

        if output_shape is None:
            output_shape = args[0].output_shape

        def evaluate(inps, **kwargs):
            inps = np.asarray(inps, dtype=np.float32)
            inps, inp_bpps = zip(*[a.evaluate(inps) for a in args])

            # Require correct shapes
            assert len(inps) == len(args)
            for i, a in zip(inps, args):
                assert endswith(i.shape, a.output_shape)

            f_value, f_backprop = f(*inps, **kwargs)
            assert endswith(f_value.shape, output_shape)
            f_value.setflags(write=False)

            def backprop(grad, *, output=None):
                if output is None:
                    output = np.zeros(self.n_params, dtype=np.float32)

                grad = np.asarray(grad, dtype=np.float32)
                assert grad.shape == f_value.shape
                inp_grads = f_backprop(grad)

                # Require correct shapes
                assert isinstance(inp_grads, tuple)
                assert len(inp_grads) >= len(inps)
                for a, b in zip(inps, inp_grads):
                    assert a.shape == b.shape

                # Average gradients in batches
                if len(f_value.shape) > len(output_shape):
                    batch = len(f_value.shape) - len(output_shape)
                    batch = np.prod(f_value.shape[:batch])
                    inp_grads = [
                        g / batch if isinstance(a, Params) else g
                        for g, a in zip(inp_grads, args)
                    ]

                pos = 0
                for a, bpp, g in zip(args, inp_bpps, inp_grads):
                    bpp(g, output=output[pos:pos+a.n_params])
                    pos += a.n_params
                assert pos == self.n_params

                return output

            return f_value, backprop

        def get_params(*, output=None):
            if output is None:
                output = np.zeros(self.n_params, dtype=np.float32)
            assert output.shape == (self.n_params,)

            pos = 0
            for a in args:
                if a.n_params >= 1:
                    a.get_params(output=output[pos:pos+a.n_params])
                    pos += a.n_params
            assert pos == self.n_params

            return output

        def load_params(new_value):
            assert new_value.shape == (self.n_params,)

            pos = 0
            for a in args:
                if a.n_params >= 1:
                    a.load_params(new_value[pos:pos+a.n_params])
                    pos += a.n_params
            assert pos == self.n_params

        self.evaluate = evaluate
        self.output_shape = output_shape
        self.n_params = sum(a.n_params for a in args)
        self.get_params = get_params
        self.load_params = load_params

        if len(self.output_shape) == 1:
            self.n_outputs = self.output_shape[0]

    def __call__(self, inps, **kwargs):
        outs, _ = self.evaluate(inps, **kwargs)
        return outs

def normed_columns(inps, outs):
    m = np.random.randn(inps, outs).astype(np.float32)
    return m / np.sqrt(np.sum(np.square(m), axis=0))

def Linear(inner, n_outputs, *, init=normed_columns):
    if not callable(init):
        init = (lambda m: lambda *s: normed_columns(*s) * m)(float(init))
    return Layer(inner, Params(inner.n_outputs, n_outputs, init=init),
        f=backprop.matmul, output_shape=(n_outputs,))

def Bias(inner, *, init=np.zeros):
    return Layer(inner, Params(*inner.output_shape, init=init),
        f=backprop.add)

def LReLU(inner, *, leak=0.1):
    return Layer(inner, f=lambda a: backprop.relu(a, leak=leak))

def Tanh(inner):
    return Layer(inner, f=backprop.tanh)

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
