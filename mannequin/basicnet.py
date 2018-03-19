
import numpy as np
import mannequin.backprop as backprop

def endswith(a, b):
    if len(a) < len(b):
        return False
    return a[len(a)-len(b):] == b

class Layer(object):
    def __init__(self, *inputs, evaluate, shape, n_local_params=0,
            get_params=None, load_params=None):
        inputs = tuple(inputs)
        shape = tuple(max(1, int(s)) for s in shape)
        n_local_params = max(0, int(n_local_params))

        def dfs(to_visit, process_node):
            visited = set()
            def visit(node):
                if node in visited:
                    return
                visited.add(node)
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
            prerequisites = []
            n_params = n_local_params

        else:
            # Traverse the computation graph
            prerequisites = []
            dfs(inputs, prerequisites.append)
            n_params = sum(l.n_local_params for l in prerequisites)

            def get_params(*, output=None):
                if output is None:
                    output = np.zeros(n_params, dtype=np.float32)
                assert output.shape == (n_params,)

                pos = 0
                for layer in prerequisites:
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
                for layer in prerequisites:
                    if layer.n_local_params >= 1:
                        layer.load_params(
                            new_value[pos:pos+layer.n_local_params]
                        )
                        pos += layer.n_local_params
                assert pos == n_params

        self.inputs = inputs
        self.evaluate = evaluate
        self.prerequisites = tuple(prerequisites)
        self.shape = shape
        self.n_params = n_params
        self.n_local_params = n_local_params
        self.get_params = get_params
        self.load_params = load_params

    def __call__(self, *args, **kwargs):
        outs, _ = self.evaluate(*args, **kwargs)
        return outs

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __str__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.shape)

    __repr__ = __str__

class Input(Layer):
    def __init__(self, *shape):
        self.last_gradient = None

        def backprop(grad, output=None):
            assert endswith(grad.shape, shape)
            grad.setflags(write=False)
            self.last_gradient = grad[:]

            if output is None:
                return []
            else:
                assert len(output) == 0
                return output

        def evaluate(array):
            array = np.asarray(array, dtype=np.float32)
            assert endswith(array.shape, shape)
            return array, backprop

        super().__init__(evaluate=evaluate, shape=shape)

class Const(Layer):
    def __init__(self, value):
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

        super().__init__(evaluate=evaluate, shape=value.shape)

class Params(Layer):
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

        super().__init__(evaluate=evaluate, shape=shape,
            n_local_params=value.size, get_params=get, load_params=load)

class Function(Layer):
    def __init__(self, *args, f, shape=None):
        assert len(args) >= 1

        if shape is None:
            shape = args[0].shape

        def local_evaluate(inps, **kwargs):
            inps = [inps[a] for a in args]

            # Require correct shapes
            for i, a in zip(inps, args):
                assert endswith(i.shape, a.shape)

            f_value, f_backprop = f(*inps, **kwargs)
            assert endswith(f_value.shape, shape)
            f_value.setflags(write=False)

            return f_value, f_backprop

        def evaluate(array, **kwargs):
            array = np.asarray(array, dtype=np.float32)
            layer_outs = {}
            layer_bpps = {}

            for layer in self.prerequisites:
                if hasattr(layer, "local_evaluate"):
                    out, bpp = layer.local_evaluate(layer_outs)
                else:
                    out, bpp = layer.evaluate(array)
                layer_outs[layer] = out
                layer_bpps[layer] = bpp

            out, bpp = self.local_evaluate(layer_outs, **kwargs)
            layer_outs[self] = out
            layer_bpps[self] = bpp

            def backprop(grad, *, output=None):
                grad = np.asarray(grad, dtype=np.float32)
                assert grad.shape == layer_outs[self].shape
                batch_shape = grad.shape[:len(grad.shape)-len(shape)]

                exec_order = self.prerequisites + (self,)
                layer_grads = {self: grad}

                for layer in reversed(exec_order):
                    if len(layer.inputs) <= 0:
                        continue

                    inp_grads = layer_bpps[layer](layer_grads[layer])
                    del layer_grads[layer]

                    assert isinstance(inp_grads, tuple)
                    assert len(inp_grads) == len(layer.inputs)

                    for g, i in zip(inp_grads, layer.inputs):
                        assert g.shape == layer_outs[i].shape

                        g_batch = g.shape[:len(g.shape)-len(i.shape)]
                        assert endswith(batch_shape, g_batch)

                        if len(g_batch) < len(batch_shape):
                            # Return batch average
                            g = g / np.prod(
                                batch_shape[:len(batch_shape)-len(g_batch)]
                            )

                        if i in layer_grads:
                            layer_grads[i] = layer_grads[i] + g
                        else:
                            layer_grads[i] = g

                if output is None:
                    output = np.zeros(self.n_params, dtype=np.float32)
                else:
                    assert output.shape == (self.n_params,)

                pos = 0
                for layer in exec_order:
                    if layer.n_local_params >= 1:
                        layer_bpps[layer](
                            layer_grads[layer],
                            output=output[pos:pos+layer.n_params]
                        )
                        pos += layer.n_params
                assert pos == self.n_params

                return output

            return layer_outs[self], backprop

        self.local_evaluate = local_evaluate

        super().__init__(*args, evaluate=evaluate, shape=shape)

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
