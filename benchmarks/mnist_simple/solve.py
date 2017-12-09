#!/usr/bin/env python3

def softmax(v):
    import numpy as np
    v = v.T
    v = np.exp(v - np.amax(v, axis=0))
    v /= np.sum(v, axis=0)
    return v.T

def accuracy(a, b):
    import numpy as np
    assert a.shape == b.shape
    assert len(a.shape) == 2
    return np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1))

def run():
    import sys
    import numpy as np

    sys.path.append("../..")
    from mannequin import Adam, Trajectory
    from mannequin.basicnet import Input, Affine, LReLU

    run_id = 0 ### 0 / 1 / 2 / 3

    data = np.load("__data.npz")
    train = Trajectory(data["train_x"], data["train_y"])
    test = Trajectory(data["test_x"], data["test_y"])

    model = Input(28 * 28)
    model = LReLU(Affine(model, 128))
    model = LReLU(Affine(model, 128))
    model = Affine(model, 10)

    opt = Adam(
        np.random.randn(model.n_params) * 0.1,
        lr=0.04
    )

    print("# type batch run_id accuracy")

    for step in range(1, 5001):
        model.load_params(opt.get_value())

        idx = np.random.choice(len(train), size=128)
        batch = train[idx]

        model_outs, backprop = model.evaluate(batch.o)
        grad = batch.a - softmax(model_outs) - 0.01 * model_outs
        opt.apply_gradient(backprop(grad))

        print("train", step, run_id, accuracy(model_outs, batch.a))

        if step % 100 == 0:
            model.load_params(opt.get_value())
            model_outs, _ = model.evaluate(test.o)

            print("test", step, run_id, accuracy(model_outs, test.a))

if __name__ == "__main__":
    run()
