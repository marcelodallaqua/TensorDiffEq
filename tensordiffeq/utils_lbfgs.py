#https://www.tensorflow.org/probability/examples/Optimizers_in_TensorFlow_Probability

import contextlib
import time
import tensorflow as tf
import tensorflow_probability as tfp

# def make_val_and_grad_fn(value_fn):
#   @functools.wraps(value_fn)
#   def val_and_grad(x):
#     return tfp.math.value_and_gradient(value_fn, x)
#   return val_and_grad


@contextlib.contextmanager
def timed_execution():
    t0 = time.time()
    yield
    dt = time.time() - t0
    print('Evaluation took: %f seconds' % dt)


def np_value(tensor):
    """Get numpy value out of possibly nested tuple of tensors."""
    if isinstance(tensor, tuple):
        return type(tensor)(*(np_value(t) for t in tensor))
    else:
        return tensor.numpy()


def run(optimizer):
    """Run an optimizer and measure it's evaluation time."""
    optimizer()  # Warmup.
    with timed_execution():
        result = optimizer()
    return np_value(result)


@tf.function
def optimize_with_lbfgs(obj_func, x0, maxiter, tolerance=1e-8):
    return tfp.optimizer.lbfgs_minimize(
        obj_func,
        initial_position=x0,
        max_iterations=maxiter,
        tolerance=tolerance)
