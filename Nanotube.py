# -*- coding: utf-8 -*-

###############################################################################
# Example Nanotube
# This program computes a*, alpha* and x* and plots
# - the diagram of the first figure
# - the two diagrams of the second figure
# - the two diagrams of the second figure
# for better numerical results choose N = 1000 (line 378, 432 and 444) and M = 10000 (line 445)
###############################################################################

import numpy as np
import sympy as sp
import math
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt


###############################################################################
# Modification of the 'lam_*' functions to 'np_*' such that, e.g. they are
# appropriate for the function 'sol' below
###############################################################################
# Definition of np_E([x_0, x_1, x_2, a, alpha])
# Input: list, output: np.float
def np_E(s):
    return lam_E(s[0], s[1], s[2], s[3], s[4])

# Definition of np_jac_E([x_0, x_1, x_2, a, alpha])
# Input: list, output: row vector, np.array
def np_jac_E(s):
    return lam_jac_E(s[0], s[1], s[2], s[3], s[4])[0]

# Definition of e_V([x_0, x_1, x_2, a, alpha])
# Input: list, output: row vector, np.array
def e_V(s):
    return lam_jac_x_E(s[0], s[1], s[2], s[3], s[4])[0]

# Definition of np_jac_x_E([x_0, x_1, x_2], [a, alpha])
# Input: list, output: row vector, np.array
def np_jac_x_E(s, t):
    return lam_jac_x_E(s[0], s[1], s[2], t[0], t[1])[0]

# Definition of np_jac_xalpha_E([x_0, x_1, x_2, alpha], [a])
# Input: list, output: row vector, np.array
def np_jac_xalpha_E(s, t):
    return lam_jac_xalpha_E(s[0], s[1], s[2], t[0], s[3])[0]

# Definition of np_hess_E([x_0, x_1, x_2, a, alpha])
# Input: list, output: matrix, np.array
def np_hess_E(s):
    return lam_hess_E(s[0], s[1], s[2], s[3], s[4])

# Definition of np_hess_x_E([x_0, x_1, x_2], [a, alpha])
# Input: list, output: matrix, np.array
def np_hess_x_E(s, t):
    return lam_hess_x_E(s[0], s[1], s[2], t[0], t[1])

# Definition of np_hess_xalpha_E([x_0, x_1, x_2, alpha], [a])
# Input: list, output: matrix, np.array
def np_hess_xalpha_E(s, t):
    return lam_hess_xalpha_E(s[0], s[1], s[2], t[0], s[3])

# np_hess_V computes the Hessian matrix of the potential V at y_0 = y_0(x, a, alpha)
# Input: See lines 83-88
# Output: a 8x2x8x2x3x3 numpy array
#         for (n, m), (r, s) in {(1, 1), (6, 1), (7, 1)} the 3x3 array np_hess_V(x, a, alpha)[(m, n)][(r, s)] correspondends to \partial_{t^n p^m}\partial_{t^r p^s}V(y_0)
def np_hess_V(x, a, alpha):
    xx = np_y0((1, 1), x, a, alpha).reshape(3)
    y = np_y0((6, 1), x, a, alpha).reshape(3)
    z = np_y0((7, 1), x, a, alpha).reshape(3)
    hess = lam_hess_V(xx[0], xx[1], xx[2], y[0], y[1], y[2], z[0], z[1], z[2])
    re = np.zeros((8, 2, 8, 2, 3, 3))
    for (g, i) in [((1, 1), 0), ((6, 1), 3), ((7, 1), 6)]:
        for (h, j) in [((1, 1), 0), ((6, 1), 3), ((7, 1), 6)]:
            re[g][h] = hess[i:i+3, j:j+3]
    return re
###############################################################################



###############################################################################
# x is a 3x1 numpy array and corresponds to the point x_0
# a is the scale factor
# alpha is the angle
# k_arr is a numpy array of real numbers and corresponds to numbers in K_{id}
# g consists of the two natural numbers g[0] and g[1] and corresponds to the element t^{g[0]}p^{g[1]} of the group G
# N is a natural number large enough, e.g. 100
###############################################################################


###############################################################################
# Definiton of functions
###############################################################################
# rot computes the linear component of an group element
# Input: See lines 83-88
# Output: a 3x3 numpy array
def rot(g, alpha):
    a = math.cos(g[0]*alpha)
    b = math.sin(g[0]*alpha)
    if g[1] == 0:
        c = np.array([[a, -b], [b, a]])
        return linalg.block_diag(c, 1)
    elif g[1] == 1:
        c = np.array([[a, b], [b, -a]])
        return linalg.block_diag(c, -1)
    else:
        print('There is an error in the definition of the function rot')

# trans computes the translation component of an group element
# Input: See lines 83-88
# Output: a 1x3 numpy array
def trans(g, a):
    return np.array([[0], [0], [g[0]*a]])

# np_y0 computes the vector $g\cdot x - x$ where the group element $g$ is dependent on $a$ and $\alpha$
# Input: See lines 83-88
# Output: a 3x1 numpy array
def np_y0(g, x, a, alpha):
    return rot(g, alpha)@x + trans(g, a) - x

# group_mul computes the multiplication of the group elements $g$ and $h$
# Input: See lines 83-88
# Output: (n, m) where n, m are natural numbers such that $gh = t^n p^m$
def group_mul(g, h):
    if g[1] == 0:
        return (g[0] + h[0], h[1])
    else:
        return (g[0] - h[0], (g[1] + h[1]) % 2)

# group_inv computes the inverse of the group element $g$
# Input: See lines 83-88
# Output: (n, m) where n, m are natural numbers such that $g^{-1} = t^n p^m$
def group_inv(g):
    if g[1] == 0:
        return (-g[0], 0)
    if g[1] == 1:
        return g
    else:
        print('There is an error in the definition of the function group_inv')

# f_V computes the function f_V
# Input: See lines 83-88
# Output: a 13x2x3x3 numpy array
#         for g in supp_V the 3x3 numpy array f_V(a)[g] corresponds to f_V(g)
def f_V(x, a, alpha):
    re = np.zeros((13, 2, 3, 3))
    hess = np_hess_V(x, a, alpha)
    for g in supp_V:
        for h in supp_V:
            in_h = group_inv(h)
            in_h_g = group_mul(in_h, g)
            b = rot(in_h, alpha)
            c = hess[h][g]
            d = rot(g, alpha)
            re[in_h_g] = re[in_h_g] + b@c@d
            re[in_h] = re[in_h] - b@c
            re[g] = re[g] - c@d
            re[(0, 0)] = re[(0, 0)] + c
    return re

# Definition of the bijection phi from R to [0,...,6]
def phi(g):
    if g in [(-1, 0), (0, 0), (1, 0), (2, 0)]:
        return g[0] + 1
    elif g in [(-1, 1), (0, 1), (1, 1)]:
        return g[0] + 5
    else:
        print('There is an error in the definition of the function phi')

# gg computes the functions g_R and g_{R,0,0}
# Input: See lines 83-88
# Output: a list with two items
#         the first item is an 4x2x21x3 numpy array and corresponds to g_R
#         the second item is an 4x2x21x3 numpy array and corresponds to g_{R,0,0}
#         for g in R the 21x3 numpy array gg(a)[0][g] corresponds to g_R(g)
#         for g in R the 21x3 numpy array gg(a)[1][g] corresponds to g_{R,0,0}(g)
def gg(x, a, alpha):
    # Definition of the basis {b_1, b_2, b_3, b_4, b_5, b_6}
    b = np.zeros((21, 6))
    # Defintion of the vector b_1, b_2 and b_3
    for g in R:
        b[3*phi(g):3*phi(g)+3, 0:3] = rot(g, alpha).T@np.identity(3)
    # Definition of the vectors b_4, b_5, b_6
    c = np.zeros((3, 3, 3))
    c[0] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    c[1] = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
    c[2] = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    for i in range(3):
        for g in R:
            b[3*phi(g):3*phi(g)+3, i+3:i+4] = rot(g, alpha).T@c[i]@np_y0(g, x, a, alpha)
    b_0 = b[:, 0:4]
    # Let u_1,...,u_k be an orthonormal basis of a subspace, and let A denote the
    # n\times k matrix whose colums are u_1,...,u_k. Then the projection matrix is AA^T.
    # Calculation of orthonormal bases q and q_00
    q, r = np.linalg.qr(b)
    q_00, r_00 = np.linalg.qr(b_0)
    p = np.identity(21) - q@q.T
    p_00 = np.identity(21) - q_00@q_00.T
    # Calculation of g_R and g_R00 with supp(g_R) = supp(g_R00) = R
    # g_R00 corresponds to g_{R,0,0}
    g_R = np.zeros((4, 2, 21, 3))
    g_R00 = np.zeros((4, 2, 21, 3))
    for g in R:
        g_R[g]  = p[:, 3*phi(g):3*phi(g)+3]
        g_R00[g]  = p_00[:, 3*phi(g):3*phi(g)+3]
    return (g_R, g_R00)

# chi computes the function \chi_k
# Input: See lines 83-88
# Output: a numpy array of complex numbers which correspond to the complex numbers \chi_k(g), k in k_arr
def chi(k_arr, g, a):
    return np.exp(math.pi*2j*trans(g, a)[2, 0]*k_arr)

# ind_chi computes the function Ind \chi_k
# Input: See lines 83-88
# Output: a (np.size(k_arr))x2x2 numpy array of complex numbers which correspond to the complex 2x2 matrices \chi_k(g), k in k_arr
def ind_chi(k_arr, g, a):
    p = (0, 1)
    if g[1] == 0:
        h = group_mul(p, group_mul(g, p))
        re1 = np.multiply.outer(chi(k_arr, g, a), np.array([[1, 0], [0, 0]]))
        re2 = np.multiply.outer(chi(k_arr, h, a), np.array([[0, 0], [0, 1]]))
        return re1 + re2 # return np.array([[chi(k_arr, g, a), 0], [0, chi(k_arr, h, a)]])
    elif g[1] == 1:
        h_one = group_mul(g, p)
        h_two = group_mul(p, g)
        re1 = np.multiply.outer(chi(k_arr, h_one, a), np.array([[0, 1], [0, 0]]))
        re2 = np.multiply.outer(chi(k_arr, h_two, a), np.array([[0, 0], [1, 0]]))
        return re1 + re2 #return np.array([[0, chi(k_arr, h_one, a)], [chi(k_arr, h_two, a), 0]])
    else:
        print('There is an error in the definition of the function ind_chi')

# fou_f_V computes the function \fourier(f_V)
# Input: See lines 83-88
# Output: a (np.size(k_arr))x6x6 numpy array of complex numbers which correspond to the 6x6 matrices \fourier(f_V)(\chi_k), k in k_arr
def fou_f_V(k_arr, x, a, alpha):
    f = f_V(x, a, alpha)
    re = np.multiply.outer(k_arr, np.zeros((6, 6), dtype=complex))
    for g in supp_f_V:
        re = re + np.kron(f[g], ind_chi(k_arr, g, a))
    return re

# Input: k_arr is a numpy array of real numbers
# Return: (fou_g_R(k_arr, x, a, alpha), fou_g_R00(k_arr, x, a, alpha))

# fou_gg computes the functions \fourier(g_R) and \fourier(g_{R,0,0})
# Input: See lines 83-88
# Output: a list with two items
#         the first item is a (np.size(k_arr))x42x6 numpy array of complex numbers which correspond to the 42x6 matrices \fourier(g_R)(\chi_k), k in k_arr
#         the second item is a (np.size(k_arr))x42x6 numpy array of complex numbers which correspond to the 42x6 matrices \fourier(g_{R,0,0})(\chi_k), k in k_arr
def fou_gg(k_arr, x, a, alpha):
    g_R, g_R00 = gg(x, a, alpha)
    re = np.multiply.outer(k_arr, np.zeros((42, 6), dtype=complex))
    re_0 = np.multiply.outer(k_arr, np.zeros((42, 6), dtype=complex))
    for g in R:
        re = re + np.kron(g_R[g], ind_chi(k_arr, g, a))
        re_0 = re_0 + np.kron(g_R00[g], ind_chi(k_arr, g, a))
    return (re, re_0)

# lambdas_min_array computes \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_R)(Ind \chi_k)) and \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_{R,0,0})(Ind \chi_k))
# Input: See lines 83-88
# Output: a real (np.size(k_arr))x2 numpy array
#         the first column corresponds to the real numbers \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_R)(Ind \chi_k)), k in k_arr
#         the second column corresponds to the real numbers \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_{R,0,0})(Ind \chi_k)), k in k_arr
def lambdas_min_array(k_arr, x, a, alpha):
    fou_f = fou_f_V(k_arr, x, a, alpha)
    fou_g_R, fou_g_R00 = fou_gg(k_arr, x, a, alpha)
    re = np.zeros((np.size(k_arr), 2))
    #eig_val = np.zeros((np.size(k_arr), 6))
    #eig_val_0 = np.zeros((np.size(k_arr), 6))
    for i in range(np.size(k_arr)):
        b = linalg.eigvalsh(fou_f[i], fou_g_R[i].conjugate().T@fou_g_R[i])
        re[i, 0] = np.min(b)        
        b = linalg.eigvalsh(fou_f[i], fou_g_R00[i].conjugate().T@fou_g_R00[i])
        re[i, 1] = np.min(b)
    return re

# k_array computes an appropriate set of k-values in K_{id}
# Input: See lines 83-88
# Output: a numpy array of N real numbers which are uniformly distributed on K_{id}
def k_array(N, a, alpha):
    if alpha < 0:
        print('error')
    if alpha > np.pi:
        print('error')
    # At k=0 and k=alpha/2/pi/a the matrix fou_g_R(k)^H*fou_g_R(k) is singular
    M = int(round(alpha/np.pi*N))
    if M<=2 or N-M<=2:
        print('error0 in k_array: N should be larger')
    buffer = 1/N/2/a/2
    if buffer > alpha/2/np.pi/a-buffer:
        print('error1 in k_array: N is to small')
    if alpha/2/np.pi/a+buffer > 1/2/a:
        print('error2 in k_array: N is to small')
    k_arr1 = np.linspace(buffer, alpha/2/np.pi/a-buffer, M)
    k_arr2 = np.linspace(alpha/2/np.pi/a+buffer, 1/2/a, N-M)
    k_arr = np.concatenate([k_arr1, k_arr2])
    return k_arr

# lambdas computes \lambda_a and \lambda_{a, 0, 0}
# Input: See lines 83-88
# Output: two numbers (an onedimensional numpy array of length two)
#         the first number corresponds to \lambda_a
#         the second number corresponds to \lambda_{a, 0, 0}
def lambdas(N, x, a, alpha):
    k_arr = k_array(N, a, alpha)
    lambdasmin = lambdas_min_array(k_arr, x, a, alpha)
    return np.amin(lambdasmin, axis=0)
###############################################################################



###############################################################################
# Definition of the potential, the energy function and their partial
# derivatives with the SymPy package for symbolic computation.
###############################################################################
n_t = sp.symbols('n_t', integer=True)
a, alpha = sp.symbols(r'a \alpha', real=True)
x_0, x_1, x_2 = sp.symbols('x_0 x_1 x_2', real=True)
y_0, y_1, y_2 = sp.symbols('y_0 y_1 y_2', real=True)
z_0, z_1, z_2 = sp.symbols('z_0 z_1 z_2', real=True)
x = sp.Matrix([[x_0], [x_1], [x_2]])
y = sp.Matrix([[y_0], [y_1], [y_2]])
z = sp.Matrix([[z_0], [z_1], [z_2]])

# Defintion of the function V = V(x, y, z)
V = (x.norm()-1)**2 + (y.norm()-1)**2 + (z.norm()-1)**2
V = V + (x.dot(y)/x.norm()/y.norm()+sp.Rational(1,2))**2
V = V + (x.dot(z)/x.norm()/z.norm()+sp.Rational(1,2))**2
V = V + (y.dot(z)/y.norm()/z.norm()+sp.Rational(1,2))**2

# Definition of the functin y0 = y0(n_t, x, a, alpha)
y0 = sp.Matrix([[sp.cos(n_t*alpha), sp.sin(n_t*alpha), 0],
                [sp.sin(n_t*alpha), -sp.cos(n_t*alpha), 0], [0, 0, -1]])
y0 = (y0-sp.eye(3))*x + sp.Matrix([[0], [0], [n_t*a]])

# Definition of the function E = E(x, a, alpha)
replacement = [(x[i], y0.subs({n_t: 1})[i]) for i in range(3)]
replacement = replacement + [(y[i], y0.subs({n_t: 6})[i]) for i in range(3)]
replacement = replacement + [(z[i], y0.subs({n_t: 7})[i]) for i in range(3)]
E = V.subs(replacement, simultaneous=True)
lam_E = sp.lambdify((x_0, x_1, x_2, a, alpha), E)

# Definition of the Jacobian matrix of E
jac_E = sp.Matrix([E]).jacobian([x_0, x_1, x_2, a, alpha])
lam_jac_E = sp.lambdify((x_0, x_1, x_2, a, alpha), jac_E)
jac_x_E = sp.Matrix([E]).jacobian([x_0, x_1, x_2])
lam_jac_x_E = sp.lambdify((x_0, x_1, x_2, a, alpha), jac_x_E)
jac_xalpha_E = sp.Matrix([E]).jacobian([x_0, x_1, x_2, alpha])
lam_jac_xalpha_E = sp.lambdify((x_0, x_1, x_2, a, alpha), jac_xalpha_E)

# Definition of the Hessian matrix of E
hess_E = sp.hessian(E, [x_0, x_1, x_2, a, alpha])
lam_hess_E = sp.lambdify((x_0, x_1, x_2, a, alpha), hess_E)
hess_x_E = sp.hessian(E, [x_0, x_1, x_2])
lam_hess_x_E = sp.lambdify((x_0, x_1, x_2, a, alpha), hess_x_E)
hess_xalpha_E = sp.hessian(E, [x_0, x_1, x_2, alpha])
lam_hess_xalpha_E = sp.lambdify((x_0, x_1, x_2, a, alpha), hess_xalpha_E)

# Definition of the Hessian matrix of V
hess_V = sp.hessian(V, [x_0, x_1, x_2, y_0, y_1, y_2, z_0, z_1, z_2])
lam_hess_V = sp.lambdify((x_0, x_1, x_2, y_0, y_1, y_2, z_0, z_1, z_2), hess_V)
###############################################################################


###############################################################################
alpha_0 = 11*math.pi/31
r = 31/math.pi/math.sqrt(3)
beta = 5*math.pi/31
###############################################################################


###############################################################################
# Plot of the diagram of the first figure
###############################################################################
# Computation of the energy E(\chi_G x_0) and norm of e_V of the (5, 1) nanotube dependent on a
N = 100 # choose the natural number N big enough for good numerical results, e.g. 1000
a_arr = np.zeros(N)
E_arr = np.zeros(N)
norm_e_V = np.zeros(N)
for i in range(N):
    a_arr[i] = 0.1 + 0.4*i/N
    x_a_1 = a_arr[i]*r*math.cos(beta)
    x_a_2 = a_arr[i]*r*math.sin(beta)
    x_a_3 = a_arr[i]*7/3
    E_arr[i] = np_E([x_a_1, x_a_2, x_a_3, a_arr[i], alpha_0])
    norm_e_V[i] = linalg.norm(e_V([x_a_1, x_a_2, x_a_3, a_arr[i], alpha_0]))

plt.rcParams['text.usetex'] = True
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.plot(a_arr, E_arr, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.plot(a_arr, norm_e_V, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.figure(1)
plt.savefig('Nanotube_1.pdf')
plt.show()
###############################################################################


###############################################################################
# Computation of a*, alpha* and x*
###############################################################################
# a_0, alpha_0 and x_a0 correspond to the parameters of the (5, 1) nanotube with planar bond lenth 1
a_0 = 3/2/math.sqrt(31)
x_a0 = [a_0*r*math.cos(beta), a_0*r*math.sin(beta), a_0*7/3]
np_x_a0 = np.asarray(x_a0).reshape((3, 1)) # x_a0 stored as a numpy array
sol = optimize.root(np_jac_E, x_a0 + [a_0, alpha_0], jac=np_hess_E, method='hybr', options={'xtol':1e-20})
aAst = sol.x[3] # aAst corresponds to a^*
alphaAst = sol.x[4] # alphaAst corresponds to \alpha^*
print('The global minimum is at a* =', aAst, ', alpha* = ', alphaAst, ' and x* =', sol.x[0:3], '.')
###############################################################################


###############################################################################
# supp_V corresponds to the support of V
supp_V = [(1, 1), (6, 1), (7, 1)]
# supp_f_V corresponds to the support of f_V, i.e. {t^{-2}, t^{-1}, t^0, t^1, t^2}
supp_f_V = [(-6, 0), (-5, 0), (-1, 0), (0, 0), (1, 0), (5, 0), (6, 0), (1, 1), (6, 1), (7, 1)]
# R corresponds to \mathcal R, i.e. {t^0, t^1, t^2}
R = [(-1, 0), (0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1)]
###############################################################################


###############################################################################
# Plot of the diagrams of the second figure
###############################################################################
N = 50 # choose the natural number N big enough for good numerical results, e.g. 1000
# Plot of the left diagram
sol = optimize.root(np_jac_x_E, x_a0, args=[aAst, alphaAst], jac=np_hess_x_E, method='hybr', options={'xtol':1e-20})
x = np.row_stack(sol.x)
k_arr = k_array(N, aAst, alphaAst)
lambdasmin = lambdas_min_array(k_arr, x, aAst, alphaAst)
plt.figure(2)
plt.plot(k_arr, lambdasmin[:, 0], k_arr, lambdasmin[:, 1])
plt.plot([alphaAst/(2*np.pi*aAst)], [0], 'o', color='tab:orange')
plt.savefig('Nanotube_2.pdf')
###############################################################################
# Plot of the right diagram
N = 25 # choose the natural number N big enough for good numerical results, e.g. 1000
M = 1000 # choose the natural number M big enough for good numerical results, e.g. 10000
a_arr = np.linspace(0.25, 0.3, N)
lambda_a_arr = np.zeros(N)
lambda_a00_arr = np.zeros(N)
for i in range(N):
    a = a_arr[i]
    sol = optimize.root(np_jac_x_E, x_a0, args=[a, alphaAst], jac=np_hess_x_E, method='hybr', options={'xtol':1e-20})
    x = np.row_stack(sol.x)
    lambda_a_arr[i], lambda_a00_arr[i] = lambdas(M, x, a, alphaAst)
    if lambda_a_arr[i]<0:
        N_cri = i+1
plt.figure(3)
plt.plot(a_arr[N_cri:N], lambda_a_arr[N_cri:N], a_arr, lambda_a00_arr)
plt.plot([aAst], [0], 'o', color='tab:orange')
plt.savefig('Nanotube_3.pdf')
plt.figure(4)
plt.plot(a_arr[0:N_cri], lambda_a_arr[0:N_cri])
plt.plot([aAst], [0], marker='o', color='tab:blue')
plt.savefig('Nanotube_3_star.pdf')
###############################################################################


###############################################################################
# Plot of the diagrams of the third figure
###############################################################################
alpha_arr = np.zeros(N)
lambda_a_arr2 = np.zeros(N)
lambda_a00_arr2 = np.zeros(N)
rel_diff = np.zeros(N)
rel_diff_0 = np.zeros(N)
for i in range(N):
    a = a_arr[i]
    sol = optimize.root(np_jac_xalpha_E, x_a0+[alpha_0], args=[a], jac=np_hess_xalpha_E, method='hybr', options={'xtol':1e-20})
    x = np.row_stack(sol.x[0:3])
    alpha_arr[i] = sol.x[3]
    lambda_a_arr2[i], lambda_a00_arr2[i] = lambdas(M, x, a, alpha_arr[i])
    if lambda_a_arr2[i]<0:
        N_cri = i+1
    rel_diff[i] = abs(lambda_a_arr[i]-lambda_a_arr2[i])/max(abs(lambda_a_arr[i]), abs(lambda_a_arr2[i]))
    rel_diff_0[i] = abs(lambda_a00_arr[i]-lambda_a00_arr2[i])/max(abs(lambda_a00_arr[i]), abs(lambda_a00_arr2[i]))
# Plot of the left diagram
plt.figure(5)
plt.plot(a_arr, alpha_arr)
plt.plot([aAst], [alphaAst], 'o', color='tab:blue')
plt.savefig('Nanotube_4.pdf')
# Plot of the right diagram
plt.figure(6)
plt.plot(a_arr[N_cri:N], rel_diff[N_cri:N], a_arr, rel_diff_0)
plt.plot([aAst], [0], 'o', color='tab:blue')
plt.savefig('Nanotube_5.pdf')
###############################################################################