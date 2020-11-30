import numpy as np
import tensorflow as tf
import pandas as pd

def sliding_window(vec, m, L):
    ''' This function takes as input a time serie along with the dimensionality
    parameters m and L in order to split it in sliding windows of length m and
    L-spaced, the vector containing the prediction target for each window is
    returned '''
    n = len(vec)
    slid_w = np.array(vec[0:m*L:L])
    targets = np.array(vec[m*L])
    for i in range(1,n-m*L,L):
        slid_w = np.vstack((slid_w,vec[i:i+m*L:L]))
        targets = np.append(targets,vec[i+m*L])
    return slid_w, targets

def dtanh(x):
    ''' Derivative of tanh(x)'''
    return 1 - np.square(np.tanh(x))

def jacobian_mat(q, m, x, alpha_1, beta_0, beta_1):
    ''' This function return the Jacobian matrix of the function f (fitted by the
    FFNN) related to the sliding window x of length m, the number of neurons q of
    the hidden-layer and FFNN weights are required '''
    J = np.zeros((m,m))
    i,j = np.indices(J.shape)
    J[i-1==j] = 1
    z = [(beta_0[j]+np.array(x).dot(beta_1[:,j])).A1 for j in range(q)]
    for u in range(m):
        J[0,u] = sum([alpha_1[j]*dtanh(z[j][0])*beta_1[u,j] for j in range(q)])
    return J

def lyapunov_coeff(q, m, L, x, alpha_0, alpha_1, beta_0, beta_1):
    ''' This function estimates the lyapunov coefficient of a certain time serie
    fitted by a FFNN with a single hidden-layer. The dimensionality parameters
    and the sliding windows are reuired together with the FFNN weights '''
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
