import tensorflow as tf
import numpy as np
from .networks import *
from .models import *
from .utils import *
from .optimizers import *
from .output import print_screen
import time
import os
from tqdm.auto import tqdm, trange
from random import random, randint
import sys

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


def fit(obj, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):
    # obj.u_model = neural_net(obj.layer_sizes)
    # obj.build_loss()
    # Can adjust batch size for collocation points, here we set it to N_f
    if batch_sz is not None:
        obj.batch_sz = batch_sz
    else:
        obj.batch_sz = obj.X_f_len
        # obj.batch_sz = len(obj.x_f)

    N_f = obj.X_f_len
    # N_f = len(obj.x_f)
    n_batches = int(N_f // obj.batch_sz)
    start_time = time.time()
    # obj.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    # obj.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)

    # these cant be tf.functions on initialization since the distributed strategy requires its own
    # graph using grad and adaptgrad, so they cant be compiled as tf.functions until we know dist/non-dist
    obj.grad = tf.function(obj.grad)
    if obj.verbose: print_screen(obj)
    print("Starting Adam training")
    # tf.profiler.experimental.start('../cache/tblogdir1')
    train_op_fn = train_op_inner(obj)
    with trange(tf_iter) as t:
        for epoch in t:
            loss_value = train_op_fn(n_batches, obj)
            # Description will be displayed on the left
            t.set_description('Adam epoch %i' % (epoch + 1))
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            if epoch % 10 == 0:
                t.set_postfix(loss=loss_value.numpy())

            if loss_value < obj.min_loss['adam']:
                # Keep the information of the best model trained (lower loss function value)
                obj.best_model['adam'] = obj.u_model  # best model
                obj.min_loss['adam'] = loss_value  # loss value
                obj.best_epoch['adam'] = epoch  # best epoch

    # tf.profiler.experimental.stop()

    # tf.profiler.experimental.start('../cache/tblogdir1')
    if newton_iter > 0:
        print("Starting L-BFGS training")
        if newton_eager:
            print("Executing eager-mode L-BFGS")
            loss_and_flat_grad = obj.get_loss_and_flat_grad()
            _, _, _, best_w, min_loss, best_epoch = eager_lbfgs(loss_and_flat_grad,
                                                                get_weights(obj.u_model),
                                                                Struct(), maxIter=newton_iter, learningRate=0.8)

            # saving best Model
            best_model = obj.u_model
            set_weights(best_model, best_w, obj.sizes_w, obj.sizes_b)
            obj.best_model['l-bfgs'] = best_model  # best model
            obj.min_loss['l-bfgs'] = min_loss  # loss value
            obj.best_epoch['l-bfgs'] = best_epoch  # best epoch

            # saving best Model
            best_model = obj.u_model
            set_weights(best_model, best_w, obj.sizes_w, obj.sizes_b)
            obj.best_model['l-bfgs'] = best_model  # best model
            obj.min_loss['l-bfgs'] = min_loss  # loss value
            obj.best_epoch['l-bfgs'] = best_epoch  # best epoch

        else:
            print("Executing graph-mode L-BFGS\n Building graph...")
            print("Warning: Depending on your CPU/GPU setup, eager-mode L-BFGS may prove faster. If the computational "
                  "graph takes a long time to build, or the computation is slow, try eager-mode L-BFGS (enabled by "
                  "default)")

            lbfgs_train(obj, newton_iter)

    # tf.profiler.experimental.stop()


    # Saving the best overall method
    if obj.min_loss['adam'] <= obj.min_loss['l-bfgs']:
        obj.min_loss['overall'] = obj.min_loss['adam']
        obj.best_epoch['overall'] = obj.best_epoch['adam']
        obj.best_model['overall'] = obj.best_model['adam']
    else:
        obj.min_loss['overall'] = obj.min_loss['l-bfgs']
        obj.best_epoch['overall'] = obj.best_epoch['l-bfgs'] + tf_iter
        obj.best_model['overall'] = obj.best_model['l-bfgs']

    # @tf.function


def lbfgs_train(obj, newton_iter):
    func = graph_lbfgs(obj.u_model, obj.update_loss)

    init_params = tf.dynamic_stitch(func.idx, obj.u_model.trainable_variables)

    lbfgs_op(func, init_params, newton_iter)


@tf.function
def lbfgs_op(func, init_params, newton_iter):
    return tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func,
        initial_position=init_params,
        max_iterations=newton_iter,
        tolerance=1e-20,
    )


def train_op_inner(obj):
    @tf.function()
    def apply_grads(n_batches, obj=obj):
        for _ in range(n_batches):
            # unstack = tf.unstack(obj.u_model.trainable_variables, axis = 2)
            obj.variables = obj.u_model.trainable_variables
            if obj.isAdaptive:
                obj.variables.extend(obj.lambdas)
                loss_value, grads = obj.grad()

                n_lambdas = len(obj.lambdas)
                graph_w = grads[:-n_lambdas]
                grads_lambda = grads[-n_lambdas:]
                grad_neg = [-x for x in grads_lambda]

                obj.tf_optimizer.apply_gradients(zip(graph_w, obj.u_model.trainable_variables))
                obj.tf_optimizer_weights.apply_gradients(zip(grad_neg, obj.lambdas))
            else:
                loss_value, grads = obj.grad()
                obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
            return loss_value

    return apply_grads


def fit_dist(obj, tf_iter, newton_iter, batch_sz=None, newton_eager=True):
    def train_epoch(dataset, STEPS):
        total_loss = 0.0
        num_batches = 0.0
        # dist_col_weights = iter(col_weights)
        dist_dataset_iterator = iter(dataset)
        for _ in range(STEPS):
            total_loss += distributed_train_step(obj, next(dist_dataset_iterator))
            num_batches += 1
        train_loss = total_loss / num_batches
        return train_loss

    def train_step(obj, inputs):
        obj.dist_X_f = inputs
        # obj.dist_col_weights = col_weights
        if obj.isAdaptive:
            obj.variables = obj.u_model.trainable_variables
            obj.dist_col_weights = tf.gather(obj.col_weights, col_idx)
            print(obj.dist_col_weights)
            obj.variables.extend([obj.u_weights, obj.dist_col_weights])
            loss_value, grads = obj.grad()
            obj.tf_optimizer.apply_gradients(zip(grads[:-2], obj.u_model.trainable_variables))
            print([grads[-2], grads[-1]])
            obj.tf_optimizer_weights.apply_gradients(
                zip([-grads[-2], -grads[-1]], [obj.u_weights, obj.dist_col_weights]))
            # TODO collocation weight splitting across replicas
            # tf.scatter_nd_add(obj.col_weights, col_idx, obj.dist_col_weights)
        else:
            obj.variables = obj.u_model.trainable_variables
            loss_value, grads = obj.grad()
            obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))
        return loss_value

    @tf.function
    def distributed_train_step(obj, dataset_inputs):
        per_replica_losses = obj.strategy.run(train_step, args=(obj, dataset_inputs))
        return obj.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

    @tf.function
    def dist_loop(obj, STEPS):
        total_loss = 0.0
        num_batches = 0.0
        # dist_col_weights = iter(col_weights)
        dist_dataset_iterator = iter(obj.train_dist_dataset)
        for _ in range(STEPS):
            total_loss += distributed_train_step(obj, next(dist_dataset_iterator))
            num_batches += 1
        train_loss = total_loss / num_batches

        return train_loss

    def train_loop(obj, tf_iter, STEPS):
        print_screen(obj)
        start_time = time.time()
        with trange(tf_iter) as t:
            for epoch in t:
                loss = dist_loop(obj, STEPS)
                t.set_description('Adam epoch %i' % (epoch + 1))
                if epoch % 10 == 0:
                    elapsed = time.time() - start_time
                    t.set_postfix(loss=loss.numpy())
                    # print('It: %d, Time: %.2f, loss: %.9f' % (epoch, elapsed, tf.get_static_value(loss)))
                    start_time = time.time()

    print("starting Adam training")
    STEPS = np.max((obj.n_batches // obj.strategy.num_replicas_in_sync, 1))
    # tf.profiler.experimental.start('../cache/tblogdir1')
    train_loop(obj, tf_iter, STEPS)
    # tf.profiler.experimental.stop()

    # l-bfgs-b optimization
    print("Starting L-BFGS training")
    # lbfgs_train(obj, newton_iter)
    # tf.profiler.experimental.stop()
