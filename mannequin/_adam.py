
class BaseTwoMoments:
    def __init__(self, value, update_rule, *,
            horizon=10,
            var_horizon=100,
            print_norm=False):
        import numpy as np
        import os
        from mannequin import RunningMean

        value = np.asarray(value, dtype=np.float64)
        value.setflags(write=False)

        running_mean = RunningMean(value.shape, horizon=horizon)
        running_var = RunningMean(value.shape, horizon=var_horizon)

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def apply_gradient(grad):
            nonlocal value

            grad = np.asarray(grad, dtype=np.float64)
            assert grad.shape == value.shape

            running_mean.update(grad)
            running_var.update(np.square(grad))

            add = update_rule(running_mean.get(), running_var.get())

            if print_norm:
                print("Update norm: %10.4f" % norm(add))

            value = value + add
            value.setflags(write=False)

        self.get_value = lambda: value
        self.apply_gradient = apply_gradient

class Adam(BaseTwoMoments):
    def __init__(self, value, *, lr, epsilon=1e-8, **params):
        import numpy as np
        lr = float(lr)
        epsilon = float(epsilon)

        def update_rule(mean, var):
            return lr * (mean / (epsilon + np.sqrt(var)))

        super().__init__(value, update_rule, **params)

class Adams(BaseTwoMoments):
    def __init__(self, value, *, lr, epsilon=1e-8, **params):
        lr = float(lr)
        epsilon = float(epsilon)

        def update_rule(mean, var):
            return lr * (mean / (epsilon + var))

        super().__init__(value, update_rule, **params)
