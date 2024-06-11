# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

This code contains the model functions used in the simulations from figure 3, figure 4
and supplementary figures 2 (a)-(d), 3 - 6.
    
Koch D, Nandan A, Ramesan G, Tyukin I, Gorban A, Koseska A (2024): 
Ghost channels and ghost cycles guiding long transients in dynamical systems
In: Physical Review Letters (forthcoming)

"""

import numpy as np

#%% elementary functions in 2D

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

#%% elementary functions in 3D

def w3d(c,V,s=1):
    x,y,z = c
    x1,x2,y1,y2,z1,z2 = V
    w = 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))*(np.tanh(s*(z-z1)) - np.tanh(s*(z-z2)))
    return w

def sys3d_lin(x0,t,p):
    x,y,z = x0
    a,b,xo,yo,zo = p
    dx = a*(x+xo)
    dy = b*(y+yo)
    dz = b*(z+zo)
    return np.array([dx,dy,dz])

def sys3d_xGhost(x0,t,p):
    x,y,z = x0
    xo,yo,zo,b,r = p
    dx = r + (x+xo)**2
    dy = b*(y+yo)
    dz = b*(z+zo)
    return np.array([dx,dy,dz])

def sys3d_yGhost(x0,t,p):
    x,y,z = x0
    xo,yo,zo,b,r = p
    dx = b*(x+xo)
    dy = r + (y+yo)**2
    dz = b*(z+zo)
    return np.array([dx,dy,dz])


#%% Horchler SHC

def connectionMatrix(alpha, beta, v):
    a1,a2,a3 = alpha
    b1,b2,b3 = beta
    v1,v2,v3 = v 
    
    return np.array([
        [a1/b1, (a1+a2)/b2, (a1-a3/v3)/b3],
        [(a2-a1/v1)/b1, a2/b2, (a2+a3)/b3],
        [(a3+a1)/b1, (a3-a2/v2)/b2, a3/b3]
        ])

def Horchler2015(x0,t,p):
    a = x0
    alpha,beta,v=p
    rho = connectionMatrix(alpha, beta, v)
    da = np.zeros(3)
    for i in range(3):
        da[i] = a[i]*(alpha[i] - np.sum(rho[i,:]*a))
    return da


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

#%% Ghost cycles

def sys_ghostCycle3D(x0,t,p):
    a,s = p
    a1,a2,a3,a4 = a
    dx = 0
    dx += w3d(x0,a1,s)*sys3d_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,0.002])
    dx += w3d(x0,a2,s)*sys3d_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,0.002])  
    dx += w3d(x0,a3,s)*(-sys3d_xGhost(x0,t,[-1.5,-1.5,-0.5,1,0.002]))
    dx += w3d(x0,a4,s)*(-sys3d_yGhost(x0,t,[-0.5,-1.5,-0.5,1,0.002]))  
    return dx

def sys_ghostCycle3D_varAlpha(x0,t,p):
    a,s,alpha = p
    a1,a2,a3,a4 = a
    dx = 0
    dx += w3d(x0,a1,s)*sys3d_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,alpha])
    dx += w3d(x0,a2,s)*sys3d_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,alpha])  
    dx += w3d(x0,a3,s)*(-sys3d_xGhost(x0,t,[-1.5,-1.5,-0.5,1,alpha]))
    dx += w3d(x0,a4,s)*(-sys3d_yGhost(x0,t,[-0.5,-1.5,-0.5,1,alpha]))  
    return dx


def sys_Farjami2021(x,t,p):
    # doi: 10.1098/rsif.2021.0442 
    g = p
    
    g1 = g; g2 = g; g3 = g
    
    b1 = 1e-5
    b2 = 1e-5
    b3 = 1e-5
    
    alpha1 = 9
    alpha2 = 9
    alpha3 = 9
    
    beta1 = 0.1
    beta2 = 0.1
    beta3 = 0.1
    
    h = 3
    
    d1 = 0.2
    d2 = 0.2
    d3 = 0.2
    
    dx1 = b1 + g1 / ((1+alpha1*(x[1]**h))*(1+beta1*(x[2]**h))) - d1*x[0]
    dx2 = b2 + g2 / ((1+alpha2*(x[2]**h))*(1+beta2*(x[0]**h))) - d2*x[1]
    dx3 = b3 + g3 / ((1+alpha3*(x[0]**h))*(1+beta3*(x[1]**h))) - d3*x[2]
    
    return np.array([dx1, dx2, dx3])

#%% Ghost/Saddle hybrid

def sys_hybrid(x0,t,p):
    a,s,alpha = p
    a1,a2,a3,a4,a5,a6,a7 = a
    ls=1.4
    dx = 0
    dx += w3d(x0,a1,s)*sys3d_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,alpha])
    dx += w3d(x0,a2,s)*sys3d_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,alpha])  
    dx += w3d(x0,a3,s)*(-sys3d_yGhost(x0,t,[-0.5,-1.5,-0.5,1,alpha]))  
    dx += w3d(x0,a4,s)*sys3d_lin(x0,t,[1,-ls,-1.5,-1.5,-0.5])
    dx += w3d(x0,a5,s)*sys3d_yGhost(x0,t,[-2.5,-1.5,-0.5,-1,alpha])  
    dx += w3d(x0,a6,s)*(-sys3d_yGhost(x0,t,[-1.5,-2.5,-0.5,1,alpha]))  
    dx += w3d(x0,a7,s)*(-sys3d_xGhost(x0,t,[-2.5,-2.5,-0.5,1,alpha]))
    return dx