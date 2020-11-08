import numpy as np
import tensorflow as tf

def der_evaluation(x, func):
    x_c = tf.constant(x)
    with tf.GradientTape() as g:
         g.watch(x_c)
         y = func(x_c)
    return g.gradient(y, x_c)

def jacobian_mat(q, m, x, alpha_1, beta_0, beta_1, act_function):
    J = np.zeros((m,m))
    i,j = np.indices(J.shape)
    J[i-1==j] = 1
    z = [beta_0[j]+beta_1[:,j].dot(x) for j in range(q)]
    for u in range(m):
        J[0,u] = sum([alpha_1[n]*der_evaluation(z[j], act_function)*beta_1[u,j] for j in range(q)])
    return J

def lyapunov_coeff(T, q, m, x, alpha_0, alpha_1, beta_0, beta_1, act_function):
    M=int(T**(2/3))
    Tm=np.identity(m)
    for t in range(M,10,-1):
        xm = x[(t-1):-1:(t-m)]
        J=jacobian_mat(q, m, xm, alpha_1, beta_0, beta_1, act_function)
        Tm=np.dot(Tm,J)
    U0=np.zeros((m,1))
    U0[0]=1
    Lx=np.dot(Tm,U0)
    Lt=(np.dot(Tm,U0)).transpose()
    F=np.dot(Lx,Lt)
    l=max(np.linalg.eigvals(F))/(2*M)
    return l
