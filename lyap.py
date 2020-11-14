import numpy as np
import tensorflow as tf

def dtanh(x):
    return 1 - np.square(np.tanh(x))

def jacobian_mat(q, m, x, alpha_1, beta_0, beta_1):
    J = np.zeros((m,m))
    i,j = np.indices(J.shape)
    J[i-1==j] = 1
    z = [(beta_0[j]+np.array(x).dot(beta_1[:,j])).A1 for j in range(q)]
    for u in range(m):
        J[0,u] = sum([alpha_1[j]*dtanh(z[j][0])*beta_1[u,j] for j in range(q)])
    return J

def lyapunov_coeff(T, q, m, x, alpha_0, alpha_1, beta_0, beta_1):
    M=int(T**(2/3))
    Tm=np.identity(m)
    for t in range(M,m,-1):
        xm = x[(t-m):(t):1]
        J=jacobian_mat(q, m, xm, alpha_1, beta_0, beta_1)
        Tm=np.dot(Tm,J)
    U0=np.zeros((m,1))
    U0[0]=1
    Lx=np.dot(Tm,U0)
    Lt=(np.dot(Tm,U0)).transpose()
    F=np.dot(Lx,Lt)
    l=np.log(max(np.linalg.eigvals(F)))/(2*M)
    return l
