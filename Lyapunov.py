# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:54:32 2020
@author: Francesco
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os

def tanh_x(x_point):
    return 1-np.tanh(x_point)**2

def Sigmoid_derivative(x):
    fun=1/(1+np.exp(x))
    y=fun*(1-fun)
    return y

def Jacobian(alpha,beta,bias,x,L,m,q,t):
    k=[]
    for i in range(0,m):
        c=[]
        for j in range(0,q):
          z = bias[j]+beta[:,j].dot(x[(t-m*L):(t):L])
          der=tanh_x(z)
          c.append(alpha[j]*der*beta[i,j])
        k.append(float(sum(c)))
    J = np.zeros((m,m))
    i,j = np.indices(J.shape)
    J[i-1==j] = 1
    arr=np.array(k)
    J[0,:]=arr
    return J

def Lyapunov (alpha,beta,bias,x,L,m,q,T):
    ### T: tempo massimo considerato
    ### L,m,q: parametri del modello
    ### x : l'intera serie temporale
    ### alfa e beta: coefficienti del modelllo presenti
    ### nel loro paper per come sono definiti nella formula a fine di pag 7
    M=int(T**(2/3))
    Tm=np.identity(m)
    for i in range(M,m,-1):
        J=Jacobian(alpha,beta,bias,x,L,m,q,i)
        Tm=np.dot(Tm,J)
    U0=np.zeros((m,1))
    U0[0]=1
    Lx=np.dot(Tm,U0)
    Lt=(np.dot(Tm,U0)).transpose()
    F=np.dot(Lx,Lt)
    l=np.log(max(np.linalg.eigvals(F)))/(2*M)
    return l
