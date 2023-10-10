# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:16:41 2023

@author: koch
"""

import numpy as np

#%% elementary functions

def w(c,A,s=1):
    x,y = c
    x1,x2,y1,y2 = A
    w = 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))
    return w

def wM(c,A,s=1):
    x,y = c
    w = 0
    for a in A:
        x1,x2,y1,y2 = a
        w += 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))
    return w

def sys_lin(x0,t,p):
    x,y = x0
    a,b,xo,yo = p
    dx = a*(x+xo)
    dy = b*(y+yo)
    return np.array([dx,dy])

def sys_constant(x0,t,p):
    x,y = x0
    a,b= p
    dx = a*np.ones(x.shape)
    dy = b*np.ones(y.shape)

    return np.array([dx,dy])

def sys_xGhost(x0,t,p):
    x,y = x0
    xo,yo,b,r = p
    dx = r + (x+xo)**2
    dy = b*(y+yo)
    return np.array([dx,dy])


#%%  4 saddle heteroclinic channel

def sys_HC4(x0,t,p):
    
    a1 = [[0,1,0,1],[1,2,1,2],[2,3,2,3],[3,4,3,4]]
    a2 = [[0,1,1,2],[0,1,2,3],[0,1,3,4],[0,1,4,5],
                [1,2,2,3],[1,2,3,4],[1,2,4,5],
                [2,3,3,4],[2,3,4,5],
                [3,4,4,5]]
    a3 = [[1,2,0,1],[2,3,0,1],[2,3,1,2],
          [3,4,0,1],[3,4,1,2],[3,4,2,3],
          [4,5,0,1],[4,5,1,2],[4,5,2,3],[4,5,3,4]]
    
    s = 5 #steepness of weighting functions
    d = 1 #saddle value

    dx = 0
    for a_ in a1:
        dx += w(x0,a_,s)*sys_lin(x0,t,[1,-d,-((a_[0]+a_[1])/2),-((a_[0]+a_[1])/2)])
    dx += wM(x0,a2,s)*sys_constant(x0,t,[-0.05,-0.05])*x0[0]+wM(x0,a3,s)*sys_constant(x0,t,[0.05,0.05]) 
    return dx

#%%  4 ghost channel

def sys_ghost4(x0,t,p):
    a1 = [[0,1,0,1],[1,2,1,2],[2,3,2,3],[3,4,3,4]]
    a2 = [[0,1,1,2],[0,1,2,3],[0,1,3,4],[0,1,4,5],
                [1,2,2,3],[1,2,3,4],[1,2,4,5],
                [2,3,3,4],[2,3,4,5],
                [3,4,4,5]]
    a3 = [[1,2,0,1],
                   [2,3,0,1],[2,3,1,2],
                   [3,4,0,1],[3,4,1,2],[3,4,2,3],
                   [4,5,0,1],[4,5,1,2],[4,5,2,3],[4,5,3,4]]
    s = 5
    dx = 0
    for a_ in a1:
        dx += w(x0,a_,s)*sys_xGhost(x0,t,[-((a_[0]+a_[1])/2),-((a_[2]+a_[3])/2),-1,0.002])
    dx += wM(x0,a2,s)*sys_constant(x0,t,[-0.05,-0.05])*x0[0]+wM(x0,a3,s)*sys_constant(x0,t,[0.05,0.05]) 
    return dx
