# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:43:08 2023

@author: koch
"""


# %matplotlib qt \\ %matplotlib inline


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint
import pandas as pd
# import plotly.express as px
# from plotly.offline import plot
# import sdeint
from numpy.polynomial.polynomial import polyfit

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions'))

import functions_v09_230403 as fun 

# import matplotlib
plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

def noBackground(ax):
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    


def distanceToPoint(xs, pt):
    d = np.array([])
    for i in range(xs.shape[1]):
        d = np.append(d, np.linalg.norm(xs[:,i]-pt))
    return d

def leavesCycle(s, cp, p2, m):
    timepts = s[0,:]
    x = s[1:,:]
    dist = distanceToPoint(x,cp)
    eps = m*np.linalg.norm(cp - p2)
    t = np.where(dist > eps)
    if t[0].size > 0:
        t_exit = timepts[np.min(t)]
    else:
        t_exit = np.inf
    return t_exit    


def RK4_na_noisy_pos(f,p,ICs,t0,dt,t_end, sigma=0, naFun = None,naFunParams = None):     # args: ODE system, parameters, initial conditions, starting time t0, dt, number of steps
        steps = int((t_end-t0)/dt)
        x = np.zeros([steps,len(ICs)])
        t = np.zeros(steps,dtype=float)
        x[0,:] = ICs
        t[0] = t0
        
        if naFun != None and naFunParams != None:
            for i in range(1,steps):
                
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1,:],t[i-1],p,naFun,naFunParams)*dt
                k2 = f(x[i-1,:]+k1/2,t[i-1],p,naFun,naFunParams)*dt
                k3 = f(x[i-1,:]+k2/2,t[i-1],p,naFun,naFunParams)*dt
                k4 = f(x[i-1,:]+k3,t[i-1],p,naFun,naFunParams)*dt
                x_next = x[i-1,:] + (k1+2*k2+2*k3+k4)/6
                dW=sigma*np.sqrt(dt)*np.random.normal() # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x_ = x_next + dW
                x_[x_<0] = 0
                x[i,:] = x_
        else:
            for i in range(1,steps):
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1,:],t[i-1],p)*dt
                k2 = f(x[i-1,:]+k1/2,t[i-1],p)*dt
                k3 = f(x[i-1,:]+k2/2,t[i-1],p)*dt
                k4 = f(x[i-1,:]+k3,t[i-1],p)*dt
                x_next = x[i-1,:] + (k1+2*k2+2*k3+k4)/6
                dW=sigma*np.sqrt(dt)*np.random.normal() # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x_ = x_next + dW
                x_[x_<0] = 0
                x[i,:] = x_

        return np.vstack((t,x.T))

def hill(x,K,nH):
    return x**nH/(x**nH + K**nH)



# Models

# Ghost cycle

def w(c,V,s=1):
    x,y,z = c
    x1,x2,y1,y2,z1,z2 = V
    w = 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))*(np.tanh(s*(z-z1)) - np.tanh(s*(z-z2)))
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
    x,y,z = x0
    xo,yo,zo,b,r = p
    dx = r + (x+xo)**2
    dy = b*(y+yo)
    dz = b*(z+zo)
    return np.array([dx,dy,dz])

def sys_yGhost(x0,t,p):
    x,y,z = x0
    xo,yo,zo,b,r = p
    dx = b*(x+xo)
    dy = r + (y+yo)**2
    dz = b*(z+zo)
    return np.array([dx,dy,dz])


def sys_ghostCycle(x0,t,p):
    a,s,r = p
    a1,a2,a3,a4 = a
    dx = 0
    
    dx += w(x0,a1,s)*sys_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,r])
    dx += w(x0,a2,s)*sys_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,r])  
    dx += w(x0,a3,s)*(-sys_xGhost(x0,t,[-1.5,-1.5,-0.5,1,r]))
    dx += w(x0,a4,s)*(-sys_yGhost(x0,t,[-0.5,-1.5,-0.5,1,r]))  
    return dx

areas = [[0,1,0,1,0,1],[1,2,0,1,0,1],[1,2,1,2,0,1],[0,1,1,2,0,1]]
steepness = 10

def sys_Farjami2021(x,t,p):
    
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
    
    # h12 = 3
    # h13 = 3
    # h23 = 3
    # h21 = 3
    # h31 = 3
    # h32 = 3
    h = 3
    
    d1 = 0.2
    d2 = 0.2
    d3 = 0.2
    
    dx1 = b1 + g1 / ((1+alpha1*(x[1]**h))*(1+beta1*(x[2]**h))) - d1*x[0]
    dx2 = b2 + g2 / ((1+alpha2*(x[2]**h))*(1+beta2*(x[0]**h))) - d2*x[1]
    dx3 = b3 + g3 / ((1+alpha3*(x[0]**h))*(1+beta3*(x[1]**h))) - d3*x[2]
    
    return np.array([dx1, dx2, dx3])

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#%% SNIC ghost cycle

stepsize = 0.1
t_end = 460   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

# r = 0.2
# ICs = [[0.20987, 0.1472, 2],[0.1472, 2, 0.20987], [2, 0.20987, 0.1472]]


# r = 0.5
# ICs = [[0.20987, 0.1472, 2],[0.1472, 2, 0.20987], [2, 0.20987, 0.1472]]


r = 1.51
ICs = [[0.20987, 0.1472, 6.5],[0.1472, 6.5, 0.20987], [6.5, 0.20987, 0.1472],[0.6,0.8,0.8]]

# r = 1.6
# ICs = [[0.20987, 0.1472, 6.5],[0.1472, 6.5, 0.20987], [6.5, 0.20987, 0.1472],[0.6,0.8,0.8]]

myFig = plt.figure(figsize=(9,9))
ax = myFig.add_subplot(1,1,1,projection='3d')

for ic in ICs:
    sim = fun.RK4_na_noisy(sys_Farjami2021,r,ic,0,stepsize,t_end, 1e-5, naFun = None,naFunParams = None) 
     
    simT,simX,simY,simZ = sim
    
    col = fun.euklideanVelocity(sim[1:,:].T, 1)
    # cmBounds = [col.min(), col.max()]
    cmBounds = [1e-6, 0.02]
    norm = plt.Normalize(cmBounds[0],cmBounds[1])
    cmap=cm.get_cmap('cool')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    
    for i in range(timesteps-1):
        ax.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=6,color=np.asarray(cmap(norm(col[i]))[0:3]))
        
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.set_zlim(0,7)

ax.set_xticks([0,2,4,6])
ax.xaxis.set_tick_params(labelsize=14)
ax.set_yticks([0,2,4,6])
ax.yaxis.set_tick_params(labelsize=14)
ax.set_zticks([0,2,4,6])
ax.zaxis.set_tick_params(labelsize=14)
 
ax.set_xlabel('$x_1$',fontsize=18)
ax.set_ylabel('$x_2$',fontsize=18)
ax.set_zlabel('$x_3$',fontsize=18)
        
noBackground(ax)
ax.view_init(43,30)
myFig.colorbar(sm)
# plt.tight_layout()


#%%flow


xmin, xmax = 0, 0.45
ymin, ymax = 0, 0.02
zmin, zmax = 5.5, 8

stepsize = 0.1
t_end = 50   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


plt.figure(dpi=200)
ax = plt.axes(projection ='3d')
noBackground(ax)

r = 1.6

ICs = []

for x in np.linspace(xmin,xmax,6):
    for y in np.linspace(ymin,ymax,2):
        for z in np.linspace(zmin,zmax,3):
            ICs.append([x,y,z])

for ic in ICs:
    sim = fun.RK4_na_noisy(sys_Farjami2021,r,ic,0,stepsize,t_end, 1e-5, naFun = None,naFunParams = None) 
     
    simT,simX,simY,simZ = sim
    
    ax.plot3D(simX,simY,simZ ,'-',color='k',lw = 1, alpha=1)
    
    i = 0
    nexti = 0
    maxi = sim.shape[1]-1
    darrow = 0.45
    count = 0
    while i < maxi and nexti < maxi:
        d = np.linalg.norm(sim[1:,i]-sim[1:,nexti])
        if np.linalg.norm(sim[1:,i]-sim[1:,nexti]) > darrow:
            count+=1
            x0,y0,z0 = sim[1:,i]
            x1,y1,z1 = sim[1:,i+1]
            arw = Arrow3D([x0,x1],[y0,y1],[z0,z1], arrowstyle="-|>", color="k", lw = 0.5, mutation_scale=13,alpha=1)
            ax.add_artist(arw)
            i = nexti 
        nexti+=1
    
ax.set_xlabel('$x1$',fontsize=18)
ax.set_ylabel('$x2$',fontsize=18)
ax.set_zlabel('$x3$',fontsize=18)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


# ax.view_init(-0,-88)
# ax.view_init(30,-45)
ax.view_init(27,55)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%             4 saddle ghost cycle                    %%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5]]


simulations = []
nruns = 30
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


#%% color coded trajectory
# 
for r in [0.002,-0.02]: #1,0.2

    t_end = 100    #

    stepsize = 0.01
    timesteps = int(t_end/stepsize)
    timepoints = np.linspace(0, t_end, timesteps+1)
    ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5]] 
    
    if r < 0.2:
        ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5],[1.1,1,0.5]]  
    
    myFig = plt.figure(figsize=(8,8))
    
    ax1 = myFig.add_subplot(1,1,1,projection='3d')
    ax1.set_xlim(0,2)
    ax1.set_ylim(0,2)
    ax1.set_zlim(0,2)
    
    for ic in ICs:
        sim = fun.RK4_na_noisy(sys_ghostCycle,[areas,steepness,r],ic,0,stepsize,t_end, 1e-4, naFun = None,naFunParams = None)  
        
        col = fun.euklideanVelocity(sim[1:,:].T, 1)
        # cmBounds = [col.min(), col.max()]
        # print(cmBounds)
        cmBounds = [1e-5, 0.005]
        norm = plt.Normalize(cmBounds[0],cmBounds[1])
        cmap=cm.get_cmap('cool')
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
        simT,simX,simY,simZ = sim
        for i in range(timesteps-1):
            ax1.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=6,color=np.asarray(cmap(norm(col[i]))[0:3]))
            
    noBackground(ax1)
    ax1.view_init(60, -43)
    myFig.colorbar(sm)
    plt.tight_layout()
    

#%%flow


xmin, xmax = 0, 0.8
ymin, ymax = 0.25, 0.75
zmin, zmax = 0.25, 0.75

stepsize = 0.1
t_end = 50   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


plt.figure(dpi=200)
ax = plt.axes(projection ='3d')
noBackground(ax)

r = -0.02

ICs = []

for x in np.linspace(xmin,xmax,7):
    for y in np.linspace(ymin,ymax,2):
        for z in np.linspace(zmin,zmax,2):
            ICs.append([x,y,z])

for ic in ICs:
    sim = fun.RK4_na_noisy(sys_ghostCycle,[areas,steepness,r],ic,0,stepsize,t_end, 1e-5, naFun = None,naFunParams = None) 
     
    simT,simX,simY,simZ = sim
    
    ax.plot3D(simX,simY,simZ ,'-',color='k',lw = 1, alpha=1)
    
    i = 0
    nexti = 0
    maxi = sim.shape[1]-1
    darrow = 0.25
    count = 0
    while i < maxi and nexti < maxi:
        d = np.linalg.norm(sim[1:,i]-sim[1:,nexti])
        if np.linalg.norm(sim[1:,i]-sim[1:,nexti]) > darrow:
            count+=1
            x0,y0,z0 = sim[1:,i]
            x1,y1,z1 = sim[1:,i+1]
            arw = Arrow3D([x0,x1],[y0,y1],[z0,z1], arrowstyle="-|>", color="k", lw = 0.5, mutation_scale=13,alpha=1)
            ax.add_artist(arw)
            i = nexti 
        nexti+=1
    
ax.set_xlabel('$x$',fontsize=18)
ax.set_ylabel('$y$',fontsize=18)
ax.set_zlabel('$z$',fontsize=18)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


ax.view_init(-0,-88)
# ax.view_init(43,-53)

