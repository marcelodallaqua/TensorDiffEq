import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *
from .fit import *
from tqdm.auto import tqdm, trange
from .output import print_screen
import math


class CollocationSolverND:
    def __init__(self, assimilate=False, verbose=True):
        self.assimilate = assimilate
        self.verbose = verbose
        self.losses = []
        self.tape = None

    def compile(self, layer_sizes, f_model, domain, bcs, Adaptive_type=0,
                dict_adaptive=None, init_weights=None, g=None, dist=False):
        """
        Args:
            layer_sizes: A list of layer sizes, can be overwritten via resetting u_model to a keras model
            f_model: PDE definition
            domain: a Domain object containing the information on the domain of the system
            bcs: a list of ICs/BCs for the problem
            Adaptive_type: string with the adaptive method
                                0 - None (no adaptive method)
                                1 - Self-adaptive (https://arxiv.org/pdf/2009.04544.pdf),
                                2 - Self-adaptive_loss with weights for the entire loss function,
                                3 - NTK (https://arxiv.org/abs/2007.14527)
            dict_adaptive: a dictionary with boollean indicating adaptive loss for every loss function
            init_weights: a dictionary with keys "residual" and "BCs". Values must be a tuple with dimension
                          equal to the number of  residuals and boundares conditions, respectively
            g: a function in terms of `lambda` for self-adaptive solving. Defaults to lambda^2
            dist: A boolean value determining whether the training will be distributed across multiple GPUs

        Returns:
            None
        """
        self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.layer_sizes = layer_sizes
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.bcs = bcs
        self.f_model = get_tf_model(f_model)
        self.g = g
        self.domain = domain
        self.dist = dist
        self.X_f_dims = tf.shape(self.domain.X_f)
        self.X_f_len = tf.slice(self.X_f_dims, [0], [1]).numpy()
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        # self.X_f_in = np.asarray(tmp)
        self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        self.u_model = neural_net(self.layer_sizes)
        self.Adaptive_type = Adaptive_type
        self.lambdas = self.dict_adaptive = self.lambdas_map = None

        if Adaptive_type == 0:  # baseline PINNs
            self.isAdaptive = False
        elif Adaptive_type == 1:  # Self-Adaptive PINNs
            self.weight_outside_sum = False
            self.isAdaptive = True
        elif Adaptive_type == 2:
            self.weight_outside_sum = True
            self.isAdaptive = True
        elif Adaptive_type == 3:
            self.weight_outside_sum = True
            self.isAdaptive = False
        else:
            raise Exception("Adaptive method invalid!")

        # TODO implement NTK method
        if Adaptive_type == 'ntk':
            raise Exception("NTK method has not been implemented yet")

        if (
                self.isAdaptive is False
                and self.dict_adaptive is not None
                and self.lambdas is not None
        ):
            raise Exception(
                "Adaptive weights are turned off but weight vectors were provided. Set the weight vectors to "
                "\"none\" to continue")

        if self.isAdaptive:
            if dict_adaptive is None or init_weights is None:
                raise Exception("Adaptive weights selected but no inputs were specified!")

            # check if at least one loss was marked to be adaptive
            is_all_false = all(not any(value) for value in dict_adaptive.values())
            if is_all_false:
                raise Exception("Adaptive method was selected but none loss was marked to be adaptive")

            self.dict_adaptive = dict_adaptive
            self.lambdas, self.lambdas_map = initialize_weights_loss(init_weights, dict_adaptive)

    def compile_data(self, x, t, y):
        if not self.assimilate:
            raise Exception(
                "Assimilate needs to be set to 'true' for data assimilation. Re-initialize CollocationSolver1D with "
                "assimilate=True.")
        self.data_x = x
        self.data_t = t
        self.data_s = y

    def update_loss(self):
        loss_epoch = {}
        ntk = {}
        loss_total = 0.0
        traces = 0.0

        # Check if adaptive is allowed
        if self.isAdaptive:
            if any(self.dict_adaptive['BCs']):
                idx_lambda_bcs = self.lambdas_map['bcs'][0]

        #####################################
        # BOUNDARIES and INIT conditions
        #####################################
        for counter_bc, bc in enumerate(self.bcs):
            if bc.isPeriodic:
                raise Exception('Boundary condition type is not acceptable')
            # initial BCs, including adaptive model
            elif bc.isInit:
                u = self.u_model(bc.input) - bc.val

            elif bc.isNeumann:
                raise Exception('Boundary condition type is not acceptable')

            elif bc.isDirichlect:
                u = self.u_model(bc.input) - bc.val
            else:
                raise Exception('Boundary condition type is not acceptable')

            # compute losses
            # Check if the current BS is adaptive
            if self.isAdaptive:
                isBC_adaptive = self.dict_adaptive["BCs"][counter_bc]
            else:
                isBC_adaptive = False

            if isBC_adaptive:
                loss_bc = MSE(u, 0., self.lambdas[idx_lambda_bcs], self.weight_outside_sum)
                idx_lambda_bcs += 1
            else:
                loss_bc = MSE(u, 0.)

            loss_total = loss_total + loss_bc
            loss_epoch[f'BC_{counter_bc}'] = loss_bc

            ## if NTK
            if self.Adaptive_type == 3:
                trace = self.ntk_jacobian(u)
                ntk[f'BC_{counter_bc}'] = {'trace': trace}
                traces = traces + trace

        #####################################
        # Residual Equations
        #####################################
        # pass thorough the forward method
        f_u_preds = self.f_model(self.u_model, *self.X_f_in)

        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(f_u_preds, tuple):
            f_u_preds = f_u_preds,

        loss_res = 0.
        for counter_res, f_u_pred in enumerate(f_u_preds):
            if self.isAdaptive:
                isRes_adaptive = self.dict_adaptive["residual"][counter_res]
                if isRes_adaptive:
                    idx_lambda_res = self.lambdas_map['residual'][0]
                    if self.g is not None:
                        loss_r = g_MSE(f_u_pred, constant(0.0), self.g(self.lambdas[idx_lambda_res]))
                    else:
                        loss_r = MSE(f_u_pred, constant(0.0), self.lambdas[idx_lambda_res], self.weight_outside_sum)
                    idx_lambda_res += 1
                else:
                    loss_r = MSE(f_u_pred, constant(0.0))
            else:
                loss_r = MSE(f_u_pred, constant(0.0))

            loss_epoch[f'Residual_{counter_res}'] = loss_r
            loss_total = loss_total + loss_r

            if self.Adaptive_type == 3:
                trace = self.ntk_jacobian(f_u_pred)
                ntk[f'Residual_{counter_res}'] = {'trace': trace}
                traces = traces + trace

        if self.Adaptive_type == 3:
            loss_total = 0.
            for item in ntk:
                lambda_ntk = traces / ntk[item]['trace']
                ntk[item].update({'lambda': lambda_ntk})
                loss_epoch[item] = lambda_ntk * loss_epoch[item]
                loss_total = loss_total + loss_epoch[item]

        loss_epoch['Total Loss'] = loss_total
        self.losses.append(loss_epoch)

        return loss_total

    def ntk_jacobian(self, y):
        #
        with self.tape.stop_recording():
            jac = self.tape.jacobian(y, self.variables)
            trace = trace_ntk(jac)
        return trace

    def ntk_jacobian2(self, y):
        # check available memory
        import psutil
        memory = dict(psutil.virtual_memory()._asdict())
        # you can calculate percentage of available memory (bytes)
        available = psutil.virtual_memory().available

        # every element in the matrix uses 8 bytes (64 bits)
        n_rows = y.shape[0]
        n_col = np.sum([np.prod(v.get_shape().as_list()) for v in self.variables])
        max_n_rows = available * .05 / n_col / 8.

        if n_rows >= max_n_rows:
            a = 0

        # fix the maximum number of lines
        chunk = 512
        n_chunks = 1
        if n_rows > chunk:
            n_chunks = math.ceil(n_rows / chunk)
        chunk_size = np.minimum(n_rows, chunk)

        trace = 0.
        for n in range(n_chunks):
            y_chunk = y[n * chunk_size:(n + 1) * chunk_size, :]
            with self.tape.stop_recording():
                start_time = time.time()
                jac = self.tape.jacobian(y_chunk, self.variables)
                print('---- Elapse time %s' % (time.time() - start_time))
                trace_chunk = trace_ntk(jac)
                trace = trace + trace_chunk
        return trace

    # @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            self.tape = tape
            loss_value = self.update_loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    def fit(self, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):
        if self.isAdaptive and (batch_sz is not None):
            raise Exception("Currently we dont support minibatching for adaptive PINNs")
        if self.dist:
            BUFFER_SIZE = len(self.X_f_in[0])
            EPOCHS = tf_iter
            # devices = ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3'],
            try:
                self.strategy = tf.distribute.MirroredStrategy()
            except:
                print(
                    "Looks like we cant find any GPUs available, or your GPUs arent responding to Tensorflow's API. If "
                    "you're receiving this in error, check that your CUDA, "
                    "CUDNN, and other GPU dependencies are installed correctly with correct versioning based on your "
                    "version of Tensorflow")

            print("Number of GPU devices: {}".format(self.strategy.num_replicas_in_sync))

            self.batch_sz = batch_sz if batch_sz is not None else len(self.X_f_in[0])
            # weights_idx = tensor(list(range(len(self.x_f))), dtype=tf.int32)
            # print(weights_idx)
            # print(tf.gather(self.col_weights, weights_idx))
            N_f = len(self.X_f_in[0])
            self.n_batches = N_f // self.batch_sz

            BATCH_SIZE_PER_REPLICA = self.batch_sz
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync

            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                self.X_f_in).batch(GLOBAL_BATCH_SIZE)

            # self.train_dataset = self.train_dataset.with_options(options)

            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

            start_time = time.time()

            with self.strategy.scope():
                self.u_model = neural_net(self.layer_sizes)
                self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
                self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
                # self.dist_col_weights = tf.Variable(tf.zeros(batch_sz), validate_shape=True)

                if self.isAdaptive:
                    # self.col_weights = tf.Variable(tf.random.uniform([self.batch_sz, 1]))
                    self.u_weights = tf.Variable(self.u_weights)

            fit_dist(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)

        else:
            fit(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)

    # L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(self):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value = self.update_loss()
            grad = tape.gradient(loss_value, self.u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def predict(self, X_star):
        # predict using concatenated data
        u_star = self.u_model(X_star)
        # split data into tuples for ND support
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        # X_star = np.asarray(tmp)
        # X_star = tuple(X_star)
        X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        f_u_star = self.f_model(self.u_model, *X_star)
        return u_star.numpy(), f_u_star.numpy()

    def save(self, path):
        self.u_model.save(path)

    def load_model(self, path, compile_model=False):
        self.u_model = tf.keras.models.load_model(path, compile=compile_model)


# WIP
# TODO Distributed Discovery Model
class DiscoveryModel():
    def compile(self, layer_sizes, f_model, X, u, var, col_weights=None):
        self.layer_sizes = layer_sizes
        self.f_model = get_tf_model(f_model)
        self.X = X
        self.u = u
        self.vars = var
        self.len_ = len(var)
        self.u_model = neural_net(self.layer_sizes)
        self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_vars = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.col_weights = col_weights
        # tmp = [np.reshape(vec, (-1,1)) for i, vec in enumerate(self.X)]
        self.X_in = tuple(X)
        # self.X_in = np.asarray(tmp).T

    # print(np.shape(self.X_in))

    @tf.function
    def loss(self):
        u_pred = self.u_model(tf.concat(self.X, 1))
        f_u_pred = self.f_model(self.u_model, self.vars, *self.X_in)
        if self.col_weights is not None:
            return MSE(u_pred, self.u) + g_MSE(f_u_pred, constant(0.0), self.col_weights ** 2)
        else:
            return MSE(u_pred, self.u) + MSE(f_u_pred, constant(0.0))

    @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    @tf.function
    def train_op(self):
        self.variables = self.u_model.trainable_variables
        len_ = self.len_
        if self.col_weights is not None:

            self.variables.extend([self.col_weights])
            self.variables.extend(self.vars)
            loss_value, grads = self.grad()
            self.tf_optimizer.apply_gradients(zip(grads[:-(len_ + 2)], self.u_model.trainable_variables))
            self.tf_optimizer_weights.apply_gradients(zip([-grads[-(len_ + 1)]], [self.col_weights]))
            self.tf_optimizer_vars.apply_gradients(zip(grads[-len_:], self.vars))
        else:
            self.variables.extend(self.vars)
            loss_value, grads = self.grad()

            self.tf_optimizer.apply_gradients(zip(grads[:-(len_ + 1)], self.u_model.trainable_variables))

            self.tf_optimizer_vars.apply_gradients(zip(grads[-len_:], self.vars))

        return loss_value

    def fit(self, tf_iter):
        self.train_loop(tf_iter)

    def train_loop(self, tf_iter):  # sourcery skip: move-assign
        start_time = time.time()
        print_screen(self, discovery_model=True)
        with trange(tf_iter) as t:
            for i in t:
                loss_value = self.train_op()
                if i % 10 == 0:
                    # elapsed = time.time() - start_time
                    # print('It: %d, Time: %.2f' % (i, elapsed))
                    # tf.print(f"loss_value: {loss_value}")
                    var = [var.numpy() for var in self.vars]
                    t.set_postfix(loss=loss_value.numpy())
                    t.set_postfix(vars=var)
                    # tf.print(f"vars estimate(s): {var}")
                    # start_time = time.time()
