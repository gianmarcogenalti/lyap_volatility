import numpy as np
import tensorflow as tf
import pandas as pd

def sliding_window(vec, m, L): 
    n = len(vec)
    slid_w = np.empty(m)
    targets = np.empty(1)
    for i in range(0,n-m*L,L):
        slid_w = np.vstack((slid_w,vec[i:i+m*L:L]))
        targets = np.append(targets,vec[i+m*L])
    return slid_w, targets

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

def lyapunov_coeff(q, m, L, x, alpha_0, alpha_1, beta_0, beta_1):
    sw, t = sliding_window(x,m,L)
    T = len(sw)
    M=int(T**(2/3))
    Tm=np.identity(m)
    for t in range(T-1,T-M-1,-1):
        J=jacobian_mat(q, m, sw[t], alpha_1, beta_0, beta_1)
        Tm=np.dot(Tm,J)
    U0=np.zeros((m,1))
    U0[0]=1
    Lx=np.dot(Tm,U0)
    Lt=(np.dot(Tm,U0)).transpose()
    F=np.dot(Lx,Lt)
    l=np.log(max(np.linalg.eigvals(F)))/(2*M)
    return l
