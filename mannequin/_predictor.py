
class SimplePredictor(object):
    def __init__(self, in_size, out_size=1, *,
            hid_layers=2, hid_size=64, batch_size=64):
        import numpy as np
        from mannequin import Adam, Trajectory
        from mannequin.basicnet import Input, Affine, Tanh

        rng = np.random.RandomState()

        model = Input(int(in_size))
        for _ in range(hid_layers):
            model = Tanh(Affine(model, int(hid_size)))
        model = Affine(model, int(out_size))

        opt = Adam(
            np.random.randn(model.n_params) * 0.1,
            epsilon=1e-5
        )
        model.load_params(opt.get_value())

        def predict(inputs):
            outs, backprop = model.evaluate(inputs)
            return outs.T[0].T if int(out_size) == 1 else outs

        def learn_batch(traj, *, lr):
            outs, backprop = model.evaluate(traj.o)
            grad = np.multiply((traj.a - outs).T, traj.r).T
            opt.apply_gradient(backprop(grad), lr=lr)
            model.load_params(opt.get_value())

        def learn(inputs, labels=None, rewards=None, *, lr=1.0):
            if labels is None:
                assert isinstance(inputs, Trajectory)
            else:
                inputs = Trajectory(inputs, labels, rewards)
            n_batches = (len(inputs) + batch_size - 1) // batch_size
            lr = float(lr) / n_batches
            for _ in range(n_batches):
                idx = rng.randint(len(inputs), size=batch_size)
                learn_batch(inputs[idx], lr=lr)

        self.predict = predict
        self.learn = learn
