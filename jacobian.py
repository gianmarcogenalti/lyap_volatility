# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:54:32 2020

@author: Francesco
"""


import numpy as np
import pandas as pd
import os

x=[0.6,0.8,0.9,0.8,0.5,0.6,0.6,0.7,0.8,0.4,0.6,0.5,0.6,0.2,0.4,0.5,0.9]
beta=np.matrix([[1,2,3,4],[2,3,5,12],[1,9,7,3]])
alpha=np.array([10,2,23,2])
L=2
m=3
t=6
j=1
q=2
k=[]


def Sigmoid_derivative(x):
    fun=1/(1+np.exp(x))
    y=fun*(1-fun)
    return y
    
def Jacobian(alpha,beta,x,L,m,q,t):
    k=[]
    for i in range(1,m+1):
        for j in range(1,q+1):
            c=[]
            z=beta[0,j]+beta[j,1:].dot(x[(t-m*L):(t):L])
            der=float(Sigmoid_derivative(z))
            c.append(alpha[j]*der*beta[j,i])
        k.append(sum(c))
    J = np.zeros((m,m))
    i,j = np.indices(J.shape)
    J[i-1==j] = 1
    arr=np.array(k)
    J[0,]=arr
    return J

def Lyapunov (alpha,beta,x,L,m,q,T):
    M=int(T**(2/3))
    Tm=np.identity(m)
    for i in range(M,m*l,-1):
        J=Jacobian(alpha,beta,x,L,m,q,M-i)
        Tm=np.dot(Tm,J)
    U0=np.zeros((m,1))
    U0[0]=1
    L=np.dot(Tm,U0)
    Lt=(np.dot(Tm,U0)).transpose()
    F=np.dot(L,Lt)
    l=max(np.linalg.eigvals(F))/(2*M)
 



