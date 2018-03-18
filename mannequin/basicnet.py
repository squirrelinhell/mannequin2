
import collections
import numpy as np
import mannequin.backprop as backprop

def endswith(a, b):
    if len(a) < len(b):
        return False
    return a[len(a)-len(b):] == b

class Layer(collections.namedtuple("Layer", " ".join((
        "inputs",
        "evaluate",
        "evaluate_order",
        "shape",
        "n_params",
        "n_local_params",
        "get_params",
        "load_params",
        "get_last_gradient",
    )))):

    def __new__(cls, *inputs, evaluate, shape, n_local_params=0,
            get_params=None, load_params=None, get_last_gradient=None):
        inputs = tuple(inputs)
        shape = tuple(max(1, int(s)) for s in shape)
        n_local_params = max(0, int(n_local_params))

        def dfs(to_visit, process_node):
            visited = set()
            def visit(node):
                if id(node) in visited:
                    return
                visited.add(id(node))
                for i in node.inputs:
                    visit(i)
                process_node(node)
            for n in to_visit:
                visit(n)

        if n_local_params > 0:
            # Composite layers cannot have parameters
            assert len(inputs) == 0
            assert get_params is not None
            assert load_params is not None
            exec_order = []
            n_params = n_local_params

        else:
            # Traverse the computation graph
            exec_order = []
            dfs(inputs, exec_order.append)
            n_params = sum(l.n_local_params for l in exec_order)

            def get_params(*, output=None):
                if output is None:
                    output = np.zeros(n_params, dtype=np.float32)
                assert output.shape == (n_params,)

                pos = 0
                for layer in exec_order:
                    if layer.n_local_params >= 1:
                        layer.get_params(
                            output=output[pos:pos+layer.n_local_params]
                        )
                        pos += layer.n_local_params
                assert pos == n_params

                return output

            def load_params(new_value):
                assert new_value.shape == (n_params,)

                pos = 0
                for layer in exec_order:
                    if layer.n_local_params >= 1:
                        layer.load_params(
                            new_value[pos:pos+layer.n_local_params]
                        )
                        pos += layer.n_local_params
                assert pos == n_params

        return super().__new__(
            cls, inputs, evaluate, tuple(exec_order),
            shape, n_params, n_local_params,
            get_params, load_params, get_last_gradient
        )

    def __call__(self, *args, **kwargs):
        outs, _ = self.evaluate(*args, **kwargs)
        return outs

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __str__(self):
        return "<Layer n_inputs=%s n_params=%d>" % (
            len(self.inputs), self.n_params
        )

    __repr__ = __str__

def Input(*shape):
    last_gradient = None

    def backprop(grad, output=None):
        nonlocal last_gradient
        assert endswith(grad.shape, shape)
        grad.setflags(write=False)
        last_gradient = grad[:]
        if output is None:
            return []
        else:
            assert len(output) == 0
            return output

    def evaluate(array):
        array = np.asarray(array, dtype=np.float32)
        assert endswith(array.shape, shape)
        return array, backprop

    return Layer(evaluate=evaluate, shape=shape,
        get_last_gradient=lambda: last_gradient)

def Const(value):
    value = np.array(value, dtype=np.float32)
    assert len(value.shape) >= 1
    value.setflags(write=False)

    def backprop(grad, output=None):
        assert endswith(grad.shape, shape)
        if output is None:
            return []
        else:
            assert len(output) == 0
            return output

    def evaluate(array):
        return value[:], backprop

    return Layer(evaluate=evaluate, shape=value.shape)

def Params(*shape, init=np.zeros):
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

    def evaluate(array):
        return value[:], backprop

    def get(*, output=None):
        if output is None:
            return value.reshape(-1)
        else:
            output[:] = value.reshape(-1)
            return output

    def load(new_value):
        value.setflags(write=True)
        value[:] = new_value.reshape(shape)
        value.setflags(write=False)

    return Layer(evaluate=evaluate, shape=shape,
        n_local_params=value.size, get_params=get, load_params=load)

def Function(*args, f, shape=None):
    assert len(args) >= 1

    if shape is None:
        shape = args[0].shape

    def evaluate(array, **kwargs):
        array = np.asarray(array, dtype=np.float32)
        inps, inp_bpps = zip(*[a.evaluate(array) for a in args])

        # Require correct shapes
        assert len(inps) == len(args)
        for i, a in zip(inps, args):
            assert endswith(i.shape, a.shape)

        f_value, f_backprop = f(*inps, **kwargs)
        assert endswith(f_value.shape, shape)
        f_value.setflags(write=False)

        def backprop(grad, *, output=None):
            if output is None:
                output = np.zeros(self.n_params, dtype=np.float32)

            grad = np.asarray(grad, dtype=np.float32)
            assert grad.shape == f_value.shape
            inp_grads = f_backprop(grad)

            # Require correct shapes
            assert isinstance(inp_grads, tuple)
            inp_grads = list(inp_grads)
            assert len(inp_grads) >= len(inps)
            for a, b in zip(inps, inp_grads):
                assert a.shape == b.shape

            # Average gradients in batches
            if len(f_value.shape) > len(shape):
                batch = f_value.shape[:len(f_value.shape)-len(shape)]

                for i, g in enumerate(inp_grads):
                    g_batch = g.shape[:len(g.shape)-len(args[i].shape)]
                    assert endswith(batch, g_batch)

                    if len(g_batch) < len(batch):
                        inp_grads[i] = g / np.prod(
                            batch[:len(batch)-len(g_batch)]
                        )

            pos = 0
            for a, bpp, g in zip(args, inp_bpps, inp_grads):
                bpp(g, output=output[pos:pos+a.n_params])
                pos += a.n_params
            assert pos == self.n_params

            return output

        return f_value, backprop

    self = Layer(*args, evaluate=evaluate, shape=shape)
    return self

def normed_columns(inps, outs):
    m = np.random.randn(inps, outs).astype(np.float32)
    return m / np.sqrt(np.sum(np.square(m), axis=0))

def Add(a, b):
    return Function(a, b, f=backprop.add)

def Multiply(a, b):
    return Function(a, b, f=backprop.multiply)

def Linear(inner, n_outputs, *, init=normed_columns):
    n_inputs, = inner.shape
    if not callable(init):
        init = (lambda m: lambda *s: normed_columns(*s) * m)(float(init))
    return Function(inner, Params(n_inputs, n_outputs, init=init),
        f=backprop.matmul, shape=(n_outputs,))

def Bias(inner, *, init=np.zeros):
    return Add(inner, Params(*inner.shape, init=init))

def LReLU(inner, *, leak=0.1):
    return Function(inner, f=lambda a: backprop.relu(a, leak=leak))

def Tanh(inner):
    return Function(inner, f=backprop.tanh)

def Affine(*args, **kwargs):
    return Bias(Linear(*args, **kwargs))
