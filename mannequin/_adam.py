
class BaseTwoMoments:
    def __init__(self, value, update_rule, *,
            horizon=10,
            var_horizon=100,
            print_norm=False,
            **global_params):
        import numpy as np
        import os
        from mannequin import RunningMean

        value = np.asarray(value, dtype=np.float64)
        value.setflags(write=False)

        running_mean = RunningMean(value.shape, horizon=horizon)
        running_var = RunningMean(value.shape, horizon=var_horizon)

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def apply_gradient(grad, **local_params):
            nonlocal value

            grad = np.asarray(grad, dtype=np.float64)
            assert grad.shape == value.shape

            running_mean.update(grad)
            running_var.update(np.square(grad))

            local_params = dict(global_params, **local_params)
            add = update_rule(
                running_mean.get(),
                running_var.get(),
                **local_params
            )

            if print_norm:
                print("Update norm: %10.4f" % norm(add))

            value = value + add
            value.setflags(write=False)

        self.get_value = lambda: value
        self.apply_gradient = apply_gradient

class Adam(BaseTwoMoments):
    def __init__(self, value, **params):
        import numpy as np

        def update_rule(mean, var, *, lr, epsilon=1e-8):
            lr = float(lr)
            epsilon = float(epsilon)
            return lr * (mean / (epsilon + np.sqrt(var)))

        super().__init__(value, update_rule, **params)

class Adams(BaseTwoMoments):
    def __init__(self, value, **params):
        def update_rule(mean, var, *, lr, epsilon=1e-8):
            lr = float(lr)
            epsilon = float(epsilon)
            return lr * (mean / (epsilon + var))

        super().__init__(value, update_rule, **params)
