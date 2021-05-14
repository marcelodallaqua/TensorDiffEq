import scipy.io
import math
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import *

Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 512)
Domain.add("t", [0.0, 1.0], 201)

N_f = 50000
Domain.generate_collocation_points(N_f)


def func_ic(x):
    return x ** 2 * np.cos(math.pi * x)


# Conditions to be considered at the boundaries for the periodic BC
def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    # u_xx = tf.gradients(u_x, x)[0]
    # u_xxx = tf.gradients(u_xx, x)[0]
    # u_xxxx = tf.gradients(u_xxx, x)[0]
    return u, u_x


init = IC(Domain, [func_ic], var=[['x']])
x_periodic = periodicBC(Domain, ['x'], [deriv_model])

BCs = [init, x_periodic]


def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    c1 = tdq.utils.constant(.0001)
    c2 = tdq.utils.constant(5.0)
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u

## Which loss functions will have adaptive weights
# "residual" should a tuple for the case of multiple residual equation
# BCS have fo follow the same order as the previously defined BCs list
dict_adaptive = {"residual": [True],
                 "BCs": [True, False]}

## Weights initialization
# dictionary with keys "residual" and "BCs". Values must be a tuple with dimension
# equal to the number of  residuals and boundares conditions, respectively
init_weigths = {"residual": [tf.random.uniform([N_f, 1])],
                "BCs": [100 * tf.random.uniform([512, 1]), None]}


layer_sizes = [2, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs, Adaptive_type='self-adaptive',
              dict_adaptive=dict_adaptive, init_weigths=init_weigths)
model.fit(tf_iter=10000, newton_iter=10000)

# Load high-fidelity data for error calculation
data = scipy.io.loadmat('AC.mat')

Exact = data['uu']
Exact_u = np.real(Exact)



x = Domain.domaindict[0]['xlinspace']
t = Domain.domaindict[1]["tlinspace"]

# create mesh for plotting

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

# forward pass through model
u_pred, f_u_pred = model.predict(X_star)

error_u = tdq.helpers.find_L2_error(u_pred, u_star)
print('Error u: %e' % (error_u))

U_pred = tdq.plotting.get_griddata(X_star, u_pred.flatten(), (X, T))
FU_pred = tdq.plotting.get_griddata(X_star, f_u_pred.flatten(), (X, T))

lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

tdq.plotting.plot_solution_domain1D(model, [x, t], ub=ub, lb=lb, Exact_u=Exact_u)
