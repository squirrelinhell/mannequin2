#!/usr/bin/env python3

def run():
    import sys
    import numpy as np

    sys.path.append("../..")
    from mannequin import Trajectory, SimplePredictor

    data = np.load("__data.npz")
    train = Trajectory(data["train_x"], data["train_y"])
    test = Trajectory(data["test_x"], data["test_y"])

    def accuracy(a, b):
        assert a.shape == b.shape
        return np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1))

    model = SimplePredictor(
        28 * 28, 10,
        hid_size=128,
        classifier=True
    )

    print("# type samples accuracy")
    for step in range(1, 5001):
        batch = train[np.random.choice(len(train), size=128)]
        model.sgd_step(batch, lr=0.03) ### 0.01 / 0.03 / 0.1

        if step % 100 == 0:
            batch = train[np.random.choice(len(train), size=5000)]
            print("train", step * 128,
                accuracy(model.predict(batch.o), batch.a))
            print("test", step * 128,
                accuracy(model.predict(test.o), test.a))

if __name__ == "__main__":
    run()
