import pcg
import numpy as np
import cvxpy as cp
import gpu_library


#### Utility functions for setting up problems, testing against cvxpy

# Class to store problem data representing the following problem
# min_x 1/2 x'Hx + g'x
# subject to: l <= Ax <= u
class QPProb():
    def __init__(self, H, g, A, l, u):
        self.H, self.g, self.A, self.l, self.u = H, g, A, l, u
        self.nx, self.nc = H.shape[0], A.shape[0]

        if len(self.g.shape) == 1:
            self.g = self.g.reshape((self.nx, 1))
        if len(self.l.shape) == 1:
            self.l = self.l.reshape((self.nc, 1))
        if len(self.u.shape) == 1:
            self.u = self.u.reshape((self.nc, 1))

    def primal_res(self, x, z):
        # print( self.A@x - z)
        return np.linalg.norm(self.A@x - z, np.inf)

    def dual_res(self, x, lamb):
        # print("DUAL RES CALC")
        # print(x)
        # print(self.H@x)
        # print(self.A.T@lamb)
        # print(self.H@x + self.A.T@lamb)
        # print(self.H@x + self.g + self.A.T@lamb)
        return np.linalg.norm(self.H@x + self.g + self.A.T@lamb, np.inf)

# Generate a random qp with equality and inequality constraints
def rand_qp(nx, n_eq, n_ineq):
  H = np.random.randn(nx, nx)
  H = H.T @ H + np.eye(nx)
  H = H + H.T

  A = np.random.randn(n_eq, nx)
  C = np.random.randn(n_ineq, nx)

  active_ineq = np.random.randn(n_ineq) > 0.5

  mu = np.random.randn(n_eq)
  lamb = (np.random.randn(n_ineq))*active_ineq

  x = np.random.randn(nx)
  b = A@x
  d = C@x - np.random.randn(n_ineq)*(~active_ineq)

  g = -H@x - A.T@mu - C.T@lamb

  x = cp.Variable(nx)
  prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, np.array(H)) + g.T@x), [A@x == b, C@x >= d])
  prob.solve()

  return QPProb(H, g, np.vstack((A, C)), np.concatenate((b, d)),
          np.concatenate((b, np.full(n_ineq, np.inf)))), x.value.reshape((nx,1))

# Solve a QPProb using cvxpy
def get_sol(prob):
    x = cp.Variable(prob.nx)
    cp_prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, np.array(prob.H)) + prob.g.T@x),
                      [prob.A@x >= prob.l[:,0], prob.A@x <= prob.u[:,0]])
    cp_prob.solve()
    return x.value.reshape((prob.nx,1))

# Generate A and B for a random controllable linear system that is underactuated
def random_linear_system(nx, nu):
    assert(nx % 2 == 0)

    A = np.random.randn(nx, nx)
    U, S, Vt = np.linalg.svd(A)
    eigs = 3*np.random.rand(nx) - 1.5
    A = U@np.diagflat(eigs)@Vt
    A = (A + A.T)/2

    B = np.random.randn(nx, nu)

    assert(check_controllability(A, B))

    return A, B

# Check the controllability of a linear system
def check_controllability(A, B):
    nx = A.shape[0]
    P = np.hstack([np.linalg.matrix_power(A, i)@B for i in range(nx)])
    return np.linalg.matrix_rank(P) == nx and np.linalg.cond(P) < 1e+10

# Setup a basic mpc qp with the given horizon and unitary quadratic cost
# The primal variables are the stacked state and controls, i.e. [u_1, x_1, u_2, x_2, ...]
def setup_mpc_qp(Ad, Bd, N):
    nx, nu = Ad.shape[0], Bd.shape[1]

    # Cost matrix, identity hessian, no cost gradient
    H = np.diagflat(np.ones((nx + nu)*N))
    g = np.zeros(((nx + nu)*N, 1))

    # Equality constraints for the dynamics, essentially x_next = Ax + Bu
    A_eq = np.zeros((nx*N, (nx + nu)*N))
    for k in range(N):
        if k == 0:
            A_eq[0:nx, 0:nx + nu] = np.hstack([Bd, -np.identity(nx)])
        else:
            A_eq[nx*k:nx*(k+1), (nx + nu)*k - nx:(nx + nu)*(k+1)] = np.hstack([Ad, Bd, -np.identity(nx)])

    # Artificial torque constraints for u (selects out u_1, u_2, etc from the stacked variables)
    A_ineq = np.kron(np.identity(N), np.hstack([np.identity(nu), np.zeros((nu, nx))]))

    # Bounds, zeros for the eq constraints, and then torque limits
    l = np.vstack([np.zeros((nx*N, 1)), -1*np.ones((A_ineq.shape[0], 1))])
    u = np.vstack([np.zeros((nx*N, 1)), 1*np.ones((A_ineq.shape[0], 1))])

    qp = QPProb(H, g, np.vstack([A_eq, A_ineq]), l, u)
    qp.N = N
    return qp

## Solver functions for implementing ADMM

# ADMM iteration which solves the following problem:
# min_x 1/2 x'Hx + g'x
# subject to: Ax = z
#             l <= z <= u
# where lamb are the dual variables
def admm_iter(prob, x, z, lamb, rho, debug=False):
    # Form matrix
    schur_mat = prob.H + 1e-6*np.identity(prob.nx) + rho*prob.A.T@prob.A

    # Update x
    kkt_lhs = -prob.g + 1e-6*x + prob.A.T@(rho*z - lamb)
    if np.linalg.norm(kkt_lhs, np.inf) < 1e-100: # Fails below 1e-162
        x = x*0
    else:
        # print(schur_mat.shape)
        # print(kkt_lhs.shape)
        # print(schur_mat)
        # return
        x = pcg.solve(schur_mat, kkt_lhs, "SS" , int(schur_mat.shape[0]/prob.N))

    # Update z
    z = np.clip(prob.A@x + 1/rho*lamb, prob.l, prob.u)

    # Update lamb
    lamb = lamb + rho*(prob.A@x - z)

    # print("X")
    # print(x)
    # print("Lambda")
    # print(lamb)
    # print("z")
    # print(z)



    # Output
    if debug:
        print("r_p: %2.2e\tr_d: %2.2e" % (primal_res, dual_res))

    return x, z, lamb, rho

# Outer loop for ADMM which checks the primal and dual residuals for convergence
def admm_solve(prob, tol=1e-3, max_iter=1000, debug=False):
    x = np.zeros((prob.nx,1))
    z = np.zeros((prob.nc,1))
    lamb = np.zeros((prob.nc,1))
    rho = 0.1

    for iter in range(max_iter):
        if debug:
            print("Iter %d\tRho: %1.2e\t" % (iter + 1, rho), end="")
        x, z, lamb, rho = admm_iter(prob, x, z, lamb, rho, debug=debug)
        primal_res = prob.primal_res(x, z)
        dual_res = prob.dual_res(x, lamb)
        # print("Iter ", iter, "Primal", primal_res, "Dual", dual_res)
        if primal_res < tol and dual_res < tol:
            print("Finished in %d iters" % (iter + 1))
            break

        # Update rho
        if primal_res > 10*dual_res:
            rho = 2*rho
        elif dual_res > 10*primal_res:
            rho = 1/2*rho
        rho = np.clip(rho, 1e-6, 1e+6)

    return x, z, lamb

# MPC test

import numpy as np
np.random.seed(10)
nx, nu, N = 2, 2, 5
A, B = random_linear_system(nx, nu)
prob = setup_mpc_qp(A, B, N)

x0 = 10*np.random.rand(nx)
prob.l[0:nx, 0] = -A@x0
prob.u[0:nx, 0] = -A@x0

# print(prob.H.shape)
# print(prob.g.shape)
# print(prob.A.shape)
# print(prob.l)
# print(prob.u)
# print(prob.nx)
# print(prob.nc)

# x_sol = get_sol(prob)
print("")
print("------------------Python OUTPUT----------------------------------")
print("")

x, z, lamb = admm_solve(prob, debug=False, tol=1e-3, max_iter=100)


main_nx = prob.nx
main_nc = prob.nc


print(list(x.reshape(main_nx, )))
print(list(lamb.reshape(main_nc, )))
print(list(z.reshape(main_nc, )))


g_H = list((prob.H.T).reshape(main_nx*main_nx, ))
g_g = list((prob.g).reshape(main_nx, ))
g_A = list((prob.A.T).reshape(main_nc*main_nx, ))
g_l = list((prob.l).reshape(main_nc, ))
g_u = list((prob.u).reshape(main_nc, ))
g_x = list((np.zeros(main_nx, )))
g_lamb = list((np.zeros(main_nc, )))
g_z = list((np.zeros(main_nc, )))
rho = 0.1

sigma = 1e-6
tol = 1e-3
max_iter = 100

print("")
print("------------------GPU OUTPUT----------------------------------")
print("")

x, lamb, z = gpu_library.admm_solve(g_H, g_g, g_A, g_l, g_u, g_x, g_lamb, g_z, rho, sigma, tol, max_iter)
print(x)
print(lamb)
print(z)