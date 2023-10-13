# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:43:08 2023

@author: koch
"""


# %matplotlib qt \\ %matplotlib inline


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from scipy.integrate import odeint
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import sdeint
from numpy.polynomial.polynomial import polyfit

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions'))

import functions_ghostPaper_v1 as fun 

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

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
    a,s = p
    a1,a2,a3,a4 = a
    dx = 0
    dx += w(x0,a1,s)*sys_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,0.002])
    dx += w(x0,a2,s)*sys_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,0.002])  
    dx += w(x0,a3,s)*(-sys_xGhost(x0,t,[-1.5,-1.5,-0.5,1,0.002]))
    dx += w(x0,a4,s)*(-sys_yGhost(x0,t,[-0.5,-1.5,-0.5,1,0.002]))  
    return dx

areas = [[0,1,0,1,0,1],[1,2,0,1,0,1],[1,2,1,2,0,1],[0,1,1,2,0,1]]
steepness = 10

#Horchler SHC


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

tcColors = ['royalblue','tomato','mediumaquamarine','mediumorchid']

#%% Horchler et al 2015

alpha = np.ones(3)*2
beta = np.ones(3)
# v = np.ones(3)*1.5
v = np.ones(3)*4
par_Horchler = [alpha, beta, v]


#%%
seed_int = 1
np.random.seed(seed_int)
print('random seed %i'%seed_int)

ICs = [[1,0,0],[0,1,0],[0,0,1]]
simulations = []
nruns = 30
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)
for s in sigma:
    print(s)
    ic = 0
    for n in range(nruns):
        ic += 1 
        if ic==3: ic = 0
        simDat = RK4_na_noisy_pos(Horchler2015,par_Horchler,ICs[ic],0,stepsize,t_end, s, naFun = None,naFunParams = None)    
        simulations.append(simDat)

simulations =  np.reshape(np.asarray(simulations),(len(sigma),nruns,4,timesteps))
np.save('simdat_Horchler2015_final.npy',simulations)


#%%
# simulations = np.load('simdat_Horchler2015_v4.npy')
simulations = np.load('simdat_Horchler2015_final.npy')

#%% plot timecourses for selected noise levels  
nruns = 30
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

tcColors = ['royalblue','tomato','mediumaquamarine','mediumorchid']

# for s in  [5,8]:#[0,3,6,9]:#range(len(sigma)):
    
myFig = plt.figure(figsize=(8.6*inCm/2,6*inCm))

# plt.suptitle('$\sigma$ ='+str(sigma[5]) )
# ax1 = myFig.add_subplot(1,2,1,projection='3d')

# ax1.plot3D([1,0,0,1],[0,1,0,0],[0,0,1,0] ,'--k', alpha=1, lw = 0.5)
# ax1.scatter([1,0,0],[0,1,0],[0,0,1], marker = 'o', color = 'grey', s = 100, alpha = 1, edgecolor='k')
# ax1.plot3D(simX,simY,simZ ,'-m', alpha=0.15,lw=2.5)
# noBackground(ax1)

ax1 = myFig.add_subplot(2,1,1)
# ax1.set_title('$\sigma = 5\\times 10^{-3}$',fontsize=12)

simT,simX,simY,simZ = simulations[5,0,:,:]

ax1.plot(simT,simX ,'-', color = tcColors[0], label='$a_{1}$',lw=1)
ax1.plot(simT,simY ,'-', color = tcColors[1],label='$a_{2}$', lw=1)
ax1.plot(simT,simZ ,'-', color = tcColors[2], label='$a_{3}$',lw=1)

ax1.set_xlabel('time (a.u.)',fontsize=10)
# ax1.set_ylabel('value (a.u.)',fontsize=10)
ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()


ax1 = myFig.add_subplot(2,1,2)
# ax1.set_title('$\sigma = 5\\times 10^{-2}$',fontsize=12)

simT,simX,simY,simZ = simulations[8,0,:,:]

ax1.plot(simT,simX ,'-', color = tcColors[0], label='$a_{1}$',lw=1)
ax1.plot(simT,simY ,'-', color = tcColors[1],label='$a_{2}$', lw=1)
ax1.plot(simT,simZ ,'-', color = tcColors[2], label='$a_{3}$',lw=1)

ax1.set_xlabel('time (a.u.)',fontsize=10)
# ax1.set_ylabel('value (a.u.)',fontsize=10)
ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.subplots_adjust(top=0.936,
bottom=0.154,
left=0.163,
right=0.942,
hspace=0.0,
wspace=0.2)

#%% plot color coded trajectories


relV = []
sigs = [5,8]
p = 5
p1 = []
p2 = []

for s in sigs:

    simT,simX,simY,simZ = simulations[s,0,:,::10]
    
    vx=np.gradient(simX,simT)
    vy=np.gradient(simY,simT)
    vz=np.gradient(simZ,simT)
    v = np.sqrt(vx**2+vy**2+vz**2)
    relV.append(v)
    
    p1.append(np.percentile(v, p))
    p2.append(np.percentile(v, 100-p))


from mpl_toolkits.mplot3d.art3d import Line3DCollection


cmBounds = [min(p1), max(p2)]


norm = plt.Normalize(cmBounds[0],cmBounds[1])
cmap=cm.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

myFig = plt.figure(figsize=(8.6*inCm,4*inCm))
# myFig.colorbar(sm)
for i in range(len(sigs)):
    print(i)
    ax =  myFig.add_subplot(1,2,1+i,projection='3d')
    # myFig.colorbar(sm)
    
    simT,simX,simY,simZ = simulations[sigs[i],0,:,::10]
    
    points3D = np.array([simX, simY, simZ]).T.reshape(-1, 1, 3)
    segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)
    cols3D = relV[i]#np.linspace(0,1,len(simT))
    
    lc = Line3DCollection(segments3D, cmap='cool',norm=norm,lw=2)
    lc.set_array(cols3D)
    lc.set_linewidth(5)
    line = ax.add_collection3d(lc)
    
    
    # noBackground(ax1)
    ax.set_ylim(-0.1,1.1)
    ax.set_xlim(-0.1,1.1)
    ax.set_zlim(-0.1,1.1)
    
    ax.set_xticks([0,0.5,1])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_yticks([0,0.5,1])
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_zticks([0,0.5,1])
    ax.zaxis.set_tick_params(labelsize=8)
    
    ax.view_init(26, 45)
    noBackground(ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.tight_layout()
    # plt.show()


#%% calculate time spend at saddles

SN1 = np.array([1,0,0])
SN2 = np.array([0,1,0])
SN3 = np.array([0,0,1])

SNs = [SN1, SN2, SN3]
M_SNT = np.zeros((len(SNs),len(sigma),nruns))
eps = 0.1 


for i in range(len(sigma)):
    print(i)
    for ii in range(nruns):
            nth = 20
            for iii in range(3):
                # print(i,ii,iii)
                # dist = fun.euklDist_TvP(simulations[iii,i,ii,::nth].T, SNs[iii])
                dist = distanceToPoint(simulations[i,ii,1:,::nth],SNs[iii])
                M_SNT[iii,i,ii] = stepsize*nth*len(dist[dist<eps])
                
#%% NEW ALGORITHM: calculate period and trapping at same time


valChecks = True

nruns = 30 
sigma =  [0.0001 ,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


SN1 = np.array([1,0,0])
SN2 = np.array([0,1,0])
SN3 = np.array([0,0,1])

SNs = [SN1, SN2, SN3]

eps = 0.1 
nth = 10

allPeriods_saddle = []
allTimesAtSaddles = []
allTimesNotAtSaddles = []

avgPeriods_saddle = []
avgTimesAtSaddles = []
avgTimesNotAtSaddles = []

stdPeriods_saddle = []
stdTimesAtSaddles = []
stdTimesNotAtSaddles = []

for s in range(len(sigma)):
    print('s=', s)
    
    #individual values all runs
    allPeriods_rs = []
    allTimesAtSaddles_rs = []
    allTimesNotAtSaddles_rs = []

    #Avg all runs
    avgPeriods_rs = []
    avgTimesAtSaddles_rs = []
    avgTimesNotAtSaddles_rs = []
    
    #SD all runs
    stdPeriods_rs = []
    stdTimesAtSaddles_rs = []
    stdTimesNotAtSaddles_rs = []
    
    
    for n in range(nruns):
        
            sim_nth = simulations[s,n,:,::nth]
        
            # Thresholding
            thrCrossed = []
            for t in range(int(timesteps/nth)):
                if sim_nth[1,t] < 0.25:
                    thrCrossed.append(1)
                else:
                    thrCrossed.append(0)
            thrCrossed = np.asarray(thrCrossed)   
             
            periodsThreshold = []
            periodsThreshold_out = []
            t = np.min(np.where(thrCrossed == 1)[0]); seq = 0; seq_out = 0
            
            # Time of crossing
            t_periods = []
            while t < int(timesteps/nth)-1:
                if thrCrossed[t] == 1:
                    if thrCrossed[t]-thrCrossed[t+1] == 0:
                        seq += 1
                    else:
                        periodsThreshold.append(seq)
                        seq = 0
                        t_periods.append(t)
                else:
                    # print('bla')
                    if thrCrossed[t]-thrCrossed[t+1] == 0:
                        seq_out += 1
                    else:
                        periodsThreshold_out.append(seq_out)
                        seq_out= 0
                        t_periods.append(t)
                t+=1
                
            # periods etc
            
            nrFullPeriods = min(len(periodsThreshold),len(periodsThreshold_out))
                    
            nth_dist = 10
            
            periods_run = []
            timesAtSaddles_run = []
            timesNotAtSaddles_run = []

            tts = []
            
            if all([s == 0, n == 0, valChecks == True]):
                plt.figure()
                
                for sn in range(3):
                  plt.plot(sim_nth[0,:], sim_nth[sn+1,:] ,'-', label='SN'+str(sn+1), color = tcColors[sn], lw=3,alpha=0.5) 
                  
                for t in range(0,2*nrFullPeriods-2,2):
                    plt.vlines(t_periods[t]*nth*stepsize,0,1,'k',lw=3)
                plt.xlim(0,300)
            
            for t in range(0,2*nrFullPeriods-2,2):
                
                t_atSaddles = 0
                
                for sn in range(3):
                    for tt in range(t_periods[t],t_periods[t+2],nth_dist):
                        dist = np.linalg.norm(sim_nth[1:,tt]-SNs[sn])
                        
                        if dist<eps:
                            t_atSaddles += 1*nth_dist
                            
                        if all([s == 0, n == 0, valChecks == True]):
                            if dist<eps:
                                plt.scatter(tt*nth*stepsize,1.1,c=tcColors[sn],alpha=1)
                                
                timesAtSaddles_run.append(t_atSaddles*nth*stepsize)
                timesNotAtSaddles_run.append((t_periods[t+2]-t_periods[t])*stepsize*nth-t_atSaddles*nth*stepsize)
                periods_run.append((t_periods[t+2]-t_periods[t])*stepsize*nth)              
                    
            allPeriods_rs.append(np.asarray(periods_run))
            allTimesAtSaddles_rs.append(np.asarray(timesAtSaddles_run))
            allTimesNotAtSaddles_rs.append(np.asarray(timesNotAtSaddles_run))
             
            avgPeriods_rs.append(np.mean(np.asarray(periods_run)))
            avgTimesAtSaddles_rs.append(np.mean(np.asarray(timesAtSaddles_run)))
            avgTimesNotAtSaddles_rs.append(np.mean(np.asarray(timesNotAtSaddles_run)))
            
            stdPeriods_rs.append(np.std(np.asarray(periods_run)))
            stdTimesAtSaddles_rs.append(np.std(np.asarray(timesAtSaddles_run)))
            stdTimesNotAtSaddles_rs.append(np.std(np.asarray(timesNotAtSaddles_run)))
            
    
    allPeriods_saddle.append(allPeriods_rs)
    allTimesAtSaddles.append(allTimesAtSaddles_rs)
    allTimesNotAtSaddles.append(allTimesNotAtSaddles_rs)

    avgPeriods_saddle.append(np.mean(np.asarray(avgPeriods_rs)))
    avgTimesAtSaddles.append(np.mean(np.asarray(avgTimesAtSaddles_rs)))
    avgTimesNotAtSaddles.append(np.mean(np.asarray(avgTimesNotAtSaddles_rs)))

    stdPeriods_saddle.append( (np.mean(np.asarray(stdPeriods_rs)**2))**0.5 )
    stdTimesAtSaddles.append( (np.mean(np.asarray(stdTimesAtSaddles_rs)**2))**0.5 )
    stdTimesNotAtSaddles.append( (np.mean(np.asarray(stdTimesNotAtSaddles_rs)**2))**0.5 )
                    
    
#%% plot period and time spend in or outside saddle
    

myFig = plt.figure(figsize=(8.6*inCm/2,4*inCm))

plt.errorbar(sigma,avgPeriods_saddle,yerr=stdPeriods_saddle,color='r',capsize=1.5,fmt='-d',ms=3,label='period',lw=1)      
plt.errorbar(sigma,avgTimesAtSaddles,yerr=stdTimesAtSaddles,color='k',capsize=1.5,fmt='-o',ms=3,label='in saddle vicinity period',lw=1)   
plt.errorbar(sigma,avgTimesNotAtSaddles,yerr=stdTimesNotAtSaddles,mfc='w',mec='k',ecolor='k',capsize=1.5,fmt=':sk',ms=3,label='not in saddle vicinity',lw=1)   

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)
# plt.ylabel('time (a.u.)',fontsize=10)
# plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)

plt.subplots_adjust(top=0.925,
bottom=0.307,
left=0.187,
right=0.981,
hspace=0.2,
wspace=0.2)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%             4  ghost cycle                          %%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5]]


simulations = []
nruns = 30
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

#%%
seed_int = 1
np.random.seed(seed_int)
print('random seed %i'%seed_int)


for s in sigma:
    print(s)
    ic = 0
    for n in range(nruns):
        ic += 1 
        if ic==4: ic = 0
        simDat = fun.RK4_na_noisy(sys_ghostCycle,[areas,steepness],ICs[ic],0,stepsize,t_end, s, naFun = None,naFunParams = None)
        simulations.append(simDat)    

simulations =  np.reshape(np.asarray(simulations),(len(sigma),nruns,4,timesteps))
np.save('simdat_Ghostcycle.npy',simulations)



#%%
simulations = np.load('simdat_Ghostcycle.npy')

#%% calculate time spend at individual ghost states

g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

Gs = [g1,g2,g3,g4]
M_SNT = np.zeros((len(Gs),len(sigma),nruns))

nth = 10
stateTCs = np.zeros((len(sigma),nruns,len(Gs)+1,int(timesteps/nth))) 

eps = 0.1 

for i in range(len(sigma)):
    print(i)
    for ii in range(nruns):
            stateTCs[i,ii,0,:] = simulations[i,ii,0,::nth]
            for iii in range(4):
                # dist = fun.euklDist_TvP(simulations[iii,i,ii,::nth].T, SNs[iii])
                dist = distanceToPoint(simulations[i,ii,1:,::nth],Gs[iii])
                stateTCs[i,ii,iii+1,:] = dist
                M_SNT[iii,i,ii] = stepsize*nth*len(dist[dist<eps])
                

        
#%% plot timecourses for selected noise levels  
nruns = 30
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

# for s in  [5,8]:#[0,3,6,9]:#range(len(sigma)):
    
myFig = plt.figure(figsize=(8.6*inCm/2,6*inCm))
# plt.suptitle('$\sigma$ ='+str(sigma[5]) )
# ax1 = myFig.add_subplot(1,2,1,projection='3d')

# ax1.plot3D([1,0,0,1],[0,1,0,0],[0,0,1,0] ,'--k', alpha=1, lw = 0.5)
# ax1.scatter([1,0,0],[0,1,0],[0,0,1], marker = 'o', color = 'grey', s = 100, alpha = 1, edgecolor='k')
# ax1.plot3D(simX,simY,simZ ,'-m', alpha=0.15,lw=2.5)
# noBackground(ax1)

ax1 = myFig.add_subplot(2,1,1)
# ax1.set_title('$\sigma = 5\\times 10^{-3}$',fontsize=16)

s = 0
n = 5

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=10)
# ax1.set_ylabel('value (a.u.)',fontsize=10)
# ax1.set_ylabel('$\Theta(|| \overline{x}(t) - G_i ||)$',fontsize=10)
ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()


ax1 = myFig.add_subplot(2,1,2)
# ax1.set_title('$\sigma = 5\\times 10^{-2}$',fontsize=16)

s = 8

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)


ax1.set_xlabel('time (a.u.)',fontsize=10)
# ax1.set_ylabel('value (a.u.)',fontsize=10)
# ax1.set_ylabel('$\Theta(|| \overline{x}(t) - G_i ||)$',fontsize=10)
ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.subplots_adjust(top=0.936,
bottom=0.154,
left=0.163,
right=0.942,
hspace=0.0,
wspace=0.2)

#%% 
relV = []
sigs = [5,8]
p = 5
p1 = []
p2 = []

for s in sigs:

    simT,simX,simY,simZ = simulations[s,0,:,::10]
    
    vx=np.gradient(simX,simT)
    vy=np.gradient(simY,simT)
    vz=np.gradient(simZ,simT)
    v = np.sqrt(vx**2+vy**2+vz**2)
    relV.append(v)
    
    p1.append(np.percentile(v, p))
    p2.append(np.percentile(v, 100-p))


from mpl_toolkits.mplot3d.art3d import Line3DCollection


cmBounds = [min(p1), max(p2)]


norm = plt.Normalize(cmBounds[0],cmBounds[1])
cmap=cm.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

myFig = plt.figure(figsize=(8.6*inCm,4*inCm))
# myFig.colorbar(sm)
for i in range(len(sigs)):
    print(i)
    ax =  myFig.add_subplot(1,2,1+i,projection='3d')
    # myFig.colorbar(sm)
    simT,simX,simY,simZ = simulations[sigs[i],0,:,::10]
    
    points3D = np.array([simX, simY, simZ]).T.reshape(-1, 1, 3)
    segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)
    cols3D = relV[i]#np.linspace(0,1,len(simT))
    
    lc = Line3DCollection(segments3D, cmap='cool',norm=norm,lw=2)
    lc.set_array(cols3D)
    lc.set_linewidth(5)
    line = ax.add_collection3d(lc)
    
    # noBackground(ax1)
    ax.set_xlim(.4,1.6)
    ax.set_ylim(.4,1.6)
    ax.set_zlim(0,1)
 
    ax.set_xticks([0,0.5,1])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_yticks([0,0.5,1])
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_zticks([0,0.5,1])
    ax.zaxis.set_tick_params(labelsize=8)
    
    ax.view_init(48, 46)
    noBackground(ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.tight_layout()
    # plt.show()
  

#%% NEW ALGORITHM: calculate period and trapping at same time

valChecks = True

nruns = 30 
sigma =  [0.0001 ,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
stepsize = 0.01
t_end = 1000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

Gs = [g1,g2,g3,g4]

eps = 0.1 
nth = 10

allPeriods_ghost = []
allTimesAtGhosts = []
allTimesNotAtGhosts = []

avgPeriods_ghost = []
avgTimesAtGhosts = []
avgTimesNotAtGhosts = []

stdPeriods_ghost = []
stdTimesAtGhosts = []
stdTimesNotAtGhosts = []

for s in range(len(sigma)):
    print('s=', s)
    
    #individual values all runs
    allPeriods_rs = []
    allTimesAtGhosts_rs = []
    allTimesNotAtGhosts_rs = []

    #Avg all runs
    avgPeriods_rs = []
    avgTimesAtGhosts_rs = []
    avgTimesNotAtGhosts_rs = []
    
    #SD all runs
    stdPeriods_rs = []
    stdTimesAtGhosts_rs = []
    stdTimesNotAtGhosts_rs = []
    
    
    for n in range(nruns):
        
            sim_nth = simulations[s,n,:,::nth]
        
            # Thresholding (based on distances for ghost cycle)
            thrCrossed = []
            for t in range(int(timesteps/nth)):
                if hill(stateTCs[s,n,1,t],0.3,-3) < 0.25:
                    thrCrossed.append(1)
                else:
                    thrCrossed.append(0)
            thrCrossed = np.asarray(thrCrossed)   
             
            periodsThreshold = []
            periodsThreshold_out = []
            t = np.min(np.where(thrCrossed == 1)[0]); seq = 0; seq_out = 0
            
            # Time of crossing
            t_periods = []
            while t < int(timesteps/nth)-1:
                if thrCrossed[t] == 1:
                    if thrCrossed[t]-thrCrossed[t+1] == 0:
                        seq += 1
                    else:
                        periodsThreshold.append(seq)
                        seq = 0
                        t_periods.append(t)
                else:
                    # print('bla')
                    if thrCrossed[t]-thrCrossed[t+1] == 0:
                        seq_out += 1
                    else:
                        periodsThreshold_out.append(seq_out)
                        seq_out= 0
                        t_periods.append(t)
                t+=1
                
            # periods etc
            
            nrFullPeriods = min(len(periodsThreshold),len(periodsThreshold_out))
                    
            nth_dist = 10
            
            periods_run = []
            timesAtGhosts_run = []
            timesNotAtGhosts_run = []
            
            tts = []
            
            if all([s == 0, n == 0, valChecks == True]):
                plt.figure()
                
                for g in range(4):
                  plt.plot(stateTCs[s,n,0,:], hill(stateTCs[s,n,g+1,:],0.3,-3) ,'-', label='G'+str(g+1), color = tcColors[g], lw=3,alpha=0.5) 
                  
                for t in range(0,2*nrFullPeriods-2,2):
                    plt.vlines(t_periods[t]*nth*stepsize,0,1,'k',lw=3)
                plt.xlim(0,300)
            
            for t in range(0,2*nrFullPeriods-2,2):
                
                t_atGhosts = 0
                
                for g in range(4):
                    for tt in range(t_periods[t],t_periods[t+2],nth_dist):
                        dist = np.linalg.norm(sim_nth[1:,tt]-Gs[g])
                        
                        if dist<eps:
                            t_atGhosts += 1*nth_dist
                            
                        if all([s == 0, n == 0, valChecks == True]):
                            if dist<eps:
                                plt.scatter(tt*nth*stepsize,1.1,c=tcColors[g],alpha=1)
                                
                timesAtGhosts_run.append(t_atGhosts*nth*stepsize)
                timesNotAtGhosts_run.append((t_periods[t+2]-t_periods[t])*stepsize*nth-t_atGhosts*nth*stepsize)
                periods_run.append((t_periods[t+2]-t_periods[t])*stepsize*nth)              
                    
            allPeriods_rs.append(np.asarray(periods_run))
            allTimesAtGhosts_rs.append(np.asarray(timesAtGhosts_run))
            allTimesNotAtGhosts_rs.append(np.asarray(timesNotAtGhosts_run))
             
            avgPeriods_rs.append(np.mean(np.asarray(periods_run)))
            avgTimesAtGhosts_rs.append(np.mean(np.asarray(timesAtGhosts_run)))
            avgTimesNotAtGhosts_rs.append(np.mean(np.asarray(timesNotAtGhosts_run)))
            
            stdPeriods_rs.append(np.std(np.asarray(periods_run)))
            stdTimesAtGhosts_rs.append(np.std(np.asarray(timesAtGhosts_run)))
            stdTimesNotAtGhosts_rs.append(np.std(np.asarray(timesNotAtGhosts_run)))
            
    
    allPeriods_ghost.append(allPeriods_rs)
    allTimesAtGhosts.append(allTimesAtGhosts_rs)
    allTimesNotAtGhosts.append(allTimesNotAtGhosts_rs)

    avgPeriods_ghost.append(np.mean(np.asarray(avgPeriods_rs)))
    avgTimesAtGhosts.append(np.mean(np.asarray(avgTimesAtGhosts_rs)))
    avgTimesNotAtGhosts.append(np.mean(np.asarray(avgTimesNotAtGhosts_rs)))

    stdPeriods_ghost.append( (np.mean(np.asarray(stdPeriods_rs)**2))**0.5 )
    stdTimesAtGhosts.append( (np.mean(np.asarray(stdTimesAtGhosts_rs)**2))**0.5 )
    stdTimesNotAtGhosts.append( (np.mean(np.asarray(stdTimesNotAtGhosts_rs)**2))**0.5 )
                    
    
#%% plot period and time spend in or outside ghost
    
# plt.figure(figsize=(4,6))
# plt.subplot(2,1,1)
# plt.errorbar(sigma,avgPeriods_ghost,yerr=stdPeriods_ghost,color='k',capsize=2,fmt='-o',ms=5)   

# plt.xscale('log')
# plt.xlabel('$\sigma$',fontsize = 20)
# plt.ylabel('period T')

# plt.subplot(2,1,2)

# plt.errorbar(sigma,avgTimesAtGhosts,yerr=stdTimesAtGhosts,color='k',capsize=2,fmt='-o',ms=5,label='in ghost vicinity')   
# plt.errorbar(sigma,avgTimesNotAtGhosts,yerr=stdTimesNotAtGhosts,mfc='w',mec='k',ecolor='k',capsize=2,fmt=':sk',ms=5,label='not in ghost vicinity')   

# plt.xscale('log')
# plt.xticks([1e-4,1e-3,1e-2,1e-1])
# plt.xlabel('$\sigma$',fontsize=17)
# plt.ylabel('time spend per period')
# plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)

# plt.tight_layout()

#both in one figure
# myFig = plt.figure(figsize=(8.6*inCm/2,5.7*inCm))

myFig = plt.figure(figsize=(8.6*inCm/2,4*inCm))
   
plt.errorbar(sigma,avgPeriods_ghost,yerr=stdPeriods_ghost,color='r',capsize=1.5,fmt='-d',ms=3,label='period',lw=1)        
plt.errorbar(sigma,avgTimesAtGhosts,yerr=stdTimesAtGhosts,color='k',capsize=1.5,fmt='-o',ms=3,label='in ghost vicinity',lw=1)   
plt.errorbar(sigma,avgTimesNotAtGhosts,yerr=stdTimesNotAtGhosts,mfc='w',mec='k',ecolor='k',capsize=1.5,fmt=':sk',ms=3,label='not in ghost vicinity',lw=1)   

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks([0,25,50,75],fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)
# plt.ylabel('time (a.u.)',fontsize=10)
# plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)

plt.subplots_adjust(top=0.925,
bottom=0.307,
left=0.187,
right=0.981,
hspace=0.2,
wspace=0.2)







#%% Adjustment of time spend at saddles/ghosts

alpha = np.ones(3)*2
beta = np.ones(3)
v = np.ones(3)*4
par_Horchler = [alpha, beta, v]
stepsize = 0.01
t_end = 2000   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


s = 0.0001

simDatGhost = fun.RK4_na_noisy(sys_ghostCycle,[areas,steepness],[0.5,0.5,0.5],0,stepsize,t_end, s, naFun = None,naFunParams = None)
simDatHorchler = RK4_na_noisy_pos(Horchler2015,par_Horchler,[1,0,0],0,stepsize,t_end, s, naFun = None,naFunParams = None)    

nth = 10
SN1 = np.array([1,0,0])
SN2 = np.array([0,1,0])
SN3 = np.array([0,0,1])

g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

SNs = [SN1, SN2, SN3]
Gs = [g1,g2,g3,g4]

timeAtGhosts = []
timeAtSNs = []

eps = 0.1


for n in range(4):
    
    # binary timecourses: determine if system is in vicinity of saddle/ghost point
    
    distGhost = distanceToPoint(simDatGhost[1:,::nth],Gs[n])
    if n < 3: distSHC = distanceToPoint(simDatHorchler[1:,::nth],SNs[n])
    
    bGhost = []
    if n < 3: bSHC = []

    for i in range(int(timesteps/nth)):
        if distGhost[i] < eps:
            bGhost.append(1)
        else:
            bGhost.append(0)
        if n < 3:    
            if distSHC[i] < eps:
                bSHC.append(1)
            else:
                bSHC.append(0)
    bGhost = np.asarray(bGhost)
    if n < 3: bSHC = np.asarray(bSHC)
    
    
    # calculate time in vicinity of saddle/ghost points
    t = 0
    seq = 0
    
    while t < int(timesteps/nth)-1:
        if bGhost[t] == 1:
            if bGhost[t]-bGhost[t+1] == 0:
                seq += 1
            else:
                timeAtGhosts.append(seq*nth*stepsize)
                seq = 0
        t+=1
    
    if n < 3: 
        t = 0
        seq = 0
        
        while t < int(timesteps/nth)-1:
            if bSHC[t] == 1:
                if bSHC[t]-bSHC[t+1] == 0:
                    seq += 1
                else:
                    timeAtSNs.append(seq*nth*stepsize)
                    seq = 0
            t+=1
                
#%%

plt.figure()

plt.subplot(1,3,1)

plt.plot(simDatHorchler[0,::nth],simDatHorchler[3,::nth],alpha=1, color = tcColors[2], label = '$a_3$')
plt.plot(simDatHorchler[0,::nth],bSHC,'--', alpha=0.85, color =tcColors[2], label = '$ || \overline{a} - SN_3 || < \\epsilon$')
    
plt.xlim(20,80)
plt.legend()#bbox_to_anchor=(1,1))
plt.xlabel('time (a.u.)')
plt.ylabel('value')

plt.subplot(1,3,2)

plt.plot(simDatGhost[0,::nth],hill(distGhost,0.3,-3), alpha=0.5, color =tcColors[3], label = '$\Theta(|| \overline{x} - G_4 ||)$')        
plt.plot(simDatGhost[0,::nth],bGhost,'--', alpha=0.85, color =tcColors[3], label = '$ || \overline{x} - G_4 || < \\epsilon $')

plt.xlim(20,80)
plt.legend()#bbox_to_anchor=(1,1))#, loc="upper left")
plt.xlabel('time (a.u.)')
plt.ylabel('value')

plt.subplot(1,3,3)
plt.scatter([np.mean(timeAtSNs[1:])], [400], marker='v', s =100, color='green', label='saddles',alpha=0.5)
plt.scatter([np.mean(timeAtGhosts[1:])], [400],marker='v', s =100, color='blue', label='ghosts',alpha=0.5)
plt.hist(timeAtSNs[1:],alpha=0.5, range=(9,15),bins=30, color='green')
plt.hist(timeAtGhosts[1:],alpha=0.5,range=(9,15),bins=30, color='blue')

plt.xlabel('time spent within $\\epsilon$-vicinity (a.u.)')
plt.ylabel('count')
plt.legend()

print(100 + 100*( - np.mean(timeAtSNs[1:]))/np.mean(timeAtGhosts[1:]))

#%% SFig Hill function

simulations = np.load('simdat_Ghostcycle.npy')

#%%
myFig = plt.figure()

ax1 = myFig.add_subplot(1,3,1)

s = 0
n = 0

ax1.plot(simulations[s,n,0,:], simulations[s,n,3,:] ,'-', label='z', color = 'mediumblue', lw=3) #cm.get_cmap('magma',5)(i)
ax1.plot(simulations[s,n,0,:], simulations[s,n,1,:] ,'-', label='x', color = 'k', lw=3) #cm.get_cmap('magma',5)(i)
ax1.plot(simulations[s,n,0,:], simulations[s,n,2,:] ,'-', label='y', color = 'crimson', lw=3) #cm.get_cmap('magma',5)(i)


ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$\overline{x}(t) = ( x(t), y(t), z(t) )^T$',fontsize=18)

# ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
# ax1.set_yticks(.5])
ax1.set_ylim(0.4,1.6)
ax1.set_xlim(0,150)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()


## Distance

ax1 = myFig.add_subplot(1,3,2)

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], stateTCs[s,n,i+1,:] ,'-', label='G'+str(i+1), color = tcColors[i], lw=3) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$|| \overline{x}(t) - G_i ||$',fontsize=18)
# ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
# ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.5)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

## Hill function applied to distance

ax1 = myFig.add_subplot(1,3,3)

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=3) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$\Theta(|| \overline{x}(t) - G_i ||)$',fontsize=18)
# ax1.set_box_aspect(1/3)
# ax1.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize=16)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)