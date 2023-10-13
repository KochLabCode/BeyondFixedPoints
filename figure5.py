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

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions'))

import functions_ghostPaper_v1 as fun 

# import matplotlib
plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

inCm = 1/2.54

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
    

# plt.rcParams['font.family'] = 'cmr10'# #cmb10 cmr10
# plt.rcParams['axes.unicode_minus'] = False

def distanceToPoint(xs, pt):
    d = np.array([])
    for i in range(xs.shape[1]):
        d = np.append(d, np.linalg.norm(xs[:,i]-pt))
    return d

def hill(x,K,nH):
    return x**nH/(x**nH + K**nH)


def w(c,V,s=1):
    x,y,z = c
    x1,x2,y1,y2,z1,z2 = V
    w = 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))*(np.tanh(s*(z-z1)) - np.tanh(s*(z-z2)))
    return w

def wM(c,A,s=1):
    x,y = c
    w = 0
    for a in A:
        x1,x2,y1,y2 = a
        w += 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))
    return w


def plot_streamline(ax,sys,parameters,t,grid,traj=None,trajColor='m', fps=None,stab=None,save=None):
                
    X=grid[0];Y=grid[1]
    
    # func = lambda u1,u2 : self.model.reaction(10,[u1,u2],self.p)
    func = lambda u1,u2 : sys([u1,u2],t,parameters)
    
    u,v=func(X,Y)
    # fig, ax = plt.subplots()
    ax.streamplot(X,Y,u,v,density=2,color=[0,0,0,0.3])
    if not traj is None:
        uL,uR=traj
        # ax.plot(uL[int(t/self.dt)-40:int(t/self.dt)],uR[int(t/self.dt)-40:int(t/self.dt)],lw=5.0,c='k')
        ax.plot(uL,uR,lw=2.0,c=trajColor)
        ax.scatter(uL[0],uR[0],marker='o',s=100,color=trajColor,edgecolors='black')
        
    if not fps is None:
        nn=len(fps)
        for i in range(nn):
            uLs,uRs=fps[i]
            if stab[i]=='unstable':
                color='red'
            else:
                color='black'
            ax.scatter(uLs,uRs,marker='o',s=100,color=color,edgecolors='black')
            



def sys_constant(x0,t,p):
    x,y = x0
    a,b= p
    dx = a*np.ones(x.shape)
    dy = b*np.ones(y.shape)

    return np.array([dx,dy])

def finishLine(x):
    return -x+8

def passesLine(s,f):
    for t in range(s.shape[1]):
        y = s[2,t]
        y_hat = f(s[1,t])
        if y > y_hat:
            return True, t
    return False, 0

def sys_lin(x0,t,p):
    x,y,z = x0
    a,b,xo,yo,zo = p
    dx = a*(x+xo)
    dy = b*(y+yo)
    dz = b*(z+zo)
    return np.array([dx,dy,dz])

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



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%             network 2            %%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

area2 = [[0,1,0,1,0,1],[1,2,0,1,0,1],
         [0,1,1,2,0,1],[1,2,1,2,0,1],[2,3,1,2,0,1],
         [1,2,2,3,0,1],[2,3,2,3,0,1]]

s = 10

def sys_network2(x0,t,p):
    a,s,r = p
    a1,a2,a3,a4,a5,a6,a7 = a
    ls=1.4
    dx = 0
    dx += w(x0,a1,s)*sys_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,r])
    dx += w(x0,a2,s)*sys_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,r])  
    dx += w(x0,a3,s)*(-sys_yGhost(x0,t,[-0.5,-1.5,-0.5,1,r]))  
    dx += w(x0,a4,s)*sys_lin(x0,t,[1,-ls,-1.5,-1.5,-0.5])
    dx += w(x0,a5,s)*sys_yGhost(x0,t,[-2.5,-1.5,-0.5,-1,r])  
    dx += w(x0,a6,s)*(-sys_yGhost(x0,t,[-1.5,-2.5,-0.5,1,r]))  
    dx += w(x0,a7,s)*(-sys_xGhost(x0,t,[-2.5,-2.5,-0.5,1,r]))
    return dx


stepsize = 0.02
t_end = 1500   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

#%% flow only

# X = np.linspace(0,3, 30)
# Y = np.linspace(0,3, 30)
# grid = np.meshgrid(X, Y)

# fig, ax = plt.subplots(figsize=(7.5,7.5))
# plot_streamline(ax,sys_network2,[area2,s] ,10, grid)
# plt.title('Network 2',fontsize=16)
# plt.xlabel('x',fontsize=16);plt.ylabel('y',fontsize=16)
# plt.xlim(0,3)
# plt.ylim(0,3)
# sigma = 0.01
# sim = fun.RK4_na_noisy(sys_network2,[area2,s],[1,1.05],0,stepsize,t_end, sigma, naFun = None,naFunParams = None)
# ax.plot(sim[1,:],sim[2,:],lw=2.5)
# ax.scatter(sim[1,0],sim[2,0],marker='o',s=60,edgecolors='black')

#%% timecourse

t_end = 400
stepsize = 0.01
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

ICs1 = [[2,2,0.5]]


ICs = [ICs1]

rs = [0.002]

velocities = []
simDat = []

for i in range(len(rs)):
    r = rs[i]
    v = []
    simIC = []
    for ic in ICs[i]:
        sim = fun.RK4_na_noisy(sys_network2,[area2,s,r],ic,0,stepsize,t_end, 1e-4, naFun = None,naFunParams = None)  
        v.append(fun.euklideanVelocity(sim[1:,:].T, 1)/stepsize)
        simIC.append(sim)
    simDat.append(simIC)
    velocities.append(v)

#%% SNIC ghost cycle toy model - analysis and plotting

# myFig = plt.figure(figsize=(16,4))
h = np.asarray([])
for i in range(len(rs)):
    v = np.asarray(velocities[i]).flatten()
    # ax = myFig.add_subplot(1,5,1+i)
    # ax.hist(v,100, weights=np.zeros_like(v) + 1. / v.size)
    h = np.concatenate((h, v),axis=0) 

# ax = myFig.add_subplot(1,5,5)
# ax.hist(h,100, weights=np.zeros_like(h) + 1. / h.size)

q = 5
q1 = np.percentile(h, q)
q2 = np.percentile(h, 100-q)
q3 = np.percentile(h, 98)
# plt.vlines([q1,q2], 0,1, color = 'r')
# plt.tight_layout()

cmBounds = [q1, q2]

print('limits for colorbar: ', cmBounds)
 
norm = plt.Normalize(cmBounds[0],cmBounds[1])
cmap=cm.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

myFig = plt.figure(figsize=(4.5*inCm,7*inCm))
axLims = [[0.4,2.7],[0.4,2.7]]
axTicks = [[0.5,1.5,2.5],[0.5,1.5,2.5]]

for i in range(len(rs)):
    ax = myFig.add_subplot(1,len(rs),1+i,projection='3d')
    for ii in  range(len(ICs[i])): #[len(ICs[i])-1]:#
        print(i,ii)
        simT,simX,simY,simZ = simDat[i][ii]
        
        d = velocities[i][ii]*stepsize
        t = 0
        t_last = 0
        while t < timesteps-1:
            if sum(d[t_last:t]) > q3*stepsize:
                ax.plot3D([simX[t_last],simX[t]],[simY[t_last],simY[t]],[simZ[t_last],simZ[t]] ,'-',lw=2, color=np.asarray(cmap(norm(velocities[i][ii][t]))[0:3]))
                t_last = t
            t+=1
    
    noBackground(ax)
    ax.view_init(39,69)
    ax.set_xlim(axLims[i])
    ax.set_ylim(axLims[i])
    ax.set_zlim(axLims[i])
    
    ax.set_xticks(axTicks[i])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_yticks(axTicks[i])
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_zticks(axTicks[i])
    ax.zaxis.set_tick_params(labelsize=8)
     
    ax.set_xlabel('$x$',fontsize=10)
    ax.set_ylabel('$y$',fontsize=10)
    ax.set_zlabel('$z$',fontsize=10)
   
  
plt.subplots_adjust(
    top=0.962,
bottom=0.052,
left=0.097,
right=0.956,
hspace=0.2,
wspace=0.2)

#%%

g1 = np.array([0.5,1.5,0.5])
g2 = np.array([0.5,0.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([2.5,1.5,0.5])
g5 = np.array([2.5,2.5,0.5])
g6 = np.array([1.5,2.5,0.5])
sn1 = np.array([1.5,1.5,0.5])

points = [g1,g2,g3,sn1,g4,g5,g6]


nth = 10
stateTCs = np.zeros((len(points)+1,int(timesteps/nth))) 

eps = 0.1 
stateTCs[0,:] = simDat[0][0][0,::nth]

for i in range(len(points)):
    dist = distanceToPoint(simDat[0][0][1:,::nth],points[i])
    stateTCs[i+1,:] = dist

                

#%% plot TC

cmap=cm.get_cmap('cool')

tcColors = ['coral','tomato','crimsom','green','']


myFig = plt.figure(figsize=(7*inCm,6*inCm))
ax1 = myFig.add_subplot(1,1,1)
for i in range(len(points)):
    if i != 3:
        # myColor = cm.get_cmap('RdYlBu',7)(i)
        myColor = cm.get_cmap('seismic',7)(i)
        lst = 'solid'
    else:
        myColor = 'black'
        lst = 'dashed'
    ax1.plot(stateTCs[0,:], hill(stateTCs[i+1,:],0.3,-3) ,linestyle=lst, label=str(i+1), color = myColor, lw=1.2) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=10)
# ax1.set_ylabel('value (a.u.)',fontsize=10)
# ax1.set_ylabel('$\Theta(|| \overline{x}(t) - G_i ||)$',fontsize=10)
ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
ax1.set_yticks([0,.5,1])
# ax1.set_ylim(0,1.1)
ax1.set_xlim(0,300)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()

   
  
plt.subplots_adjust(
    top=0.962,
bottom=0.052,
left=0.097,
right=0.956,
hspace=0.2,
wspace=0.2)

