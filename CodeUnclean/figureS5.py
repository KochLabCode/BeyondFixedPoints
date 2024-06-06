# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:43:08 2023
@author: Daniel Koch

This code reproduces the results shown in figure 4 from the study:
    
Koch D, Nandan A, Ramesan G, Koseska A (2023): 
Beyond fixed points: transient quasi-stable dynamics emerging from ghost channels and cycles. 
In: Arxiv. https://doi.org/10.48550/arXiv.2309.17201


IMPORTANT:
    The files "functions.py" and "models.py" need to be in the same folder as this script.
    Running the script for the first time can take a long time. 
    To load a previous simulation, set "loadData = True"
"""

loadData = False

# Import packages etc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import functions as fun 
import models
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 



# settings for plotting
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

inCm = 1/2.54 # conversion factor inches to cm

def noBackground(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    
tcColors = ['royalblue','tomato','mediumaquamarine','mediumorchid']

def w3d(c,V,s=1):
    x,y,z = c
    x1,x2,y1,y2,z1,z2 = V
    w = 1/4*(np.tanh(s*(x-x1)) - np.tanh(s*(x-x2)))*(np.tanh(s*(y-y1)) - np.tanh(s*(y-y2)))*(np.tanh(s*(z-z1)) - np.tanh(s*(z-z2)))
    return w

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


def sys_ghostCycle3D(x0,t,p):
    a,s = p
    a1,a2,a3,a4 = a
    dx = 0
    dx += w3d(x0,a1,s)*sys3d_xGhost(x0,t,[-0.5,-0.5,-0.5,-1,-0.002])
    dx += w3d(x0,a2,s)*sys3d_yGhost(x0,t,[-1.5,-0.5,-0.5,-1,-0.002])  
    dx += w3d(x0,a3,s)*(-sys3d_xGhost(x0,t,[-1.5,-1.5,-0.5,1,-0.002]))
    dx += w3d(x0,a4,s)*(-sys3d_yGhost(x0,t,[-0.5,-1.5,-0.5,1,-0.002]))  
    return dx


#%% attractor hopping

t_end = 2000
stepsize = 0.01
timesteps = int(t_end/stepsize)
nruns = 30 # number of repetitions
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2] # noise levels
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Simulations metastable attractor hopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

# loadData = True

# set random seed (optional)
seed_int = 1
np.random.seed(seed_int)

# model parameters
areas = [[0,1,0,1,0,1],[1,2,0,1,0,1],[1,2,1,2,0,1],[0,1,1,2,0,1]]
steepness = 10

# simulation settings
ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5]] # initial conditions
t_end = 2000
stepsize = 0.01
timesteps = int(t_end/stepsize)
nruns = 5 # number of repetitions
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2] # noise levels

if loadData == False:
    # run and save simulations

    simulations = []
    for s in sigma:
        print(s)
        ic = 0
        for n in range(nruns):
            ic += 1 
            if ic==4: ic = 0
            simDat = fun.RK4_na_noisy(sys_ghostCycle3D,[areas,steepness],ICs[ic],0,stepsize,t_end, s, naFun = None,naFunParams = None)
            simulations.append(simDat)    
    
    simulations =  np.reshape(np.asarray(simulations),(len(sigma),nruns,4,timesteps))
    np.save('simdat_metastableAttractors.npy',simulations)

elif loadData == True:
    simulations = np.load('simdat_metastableAttractors.npy') 


        
#%% Figure 4 (f) - Timecourses for selected noise levels  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def distanceToPoint(xs, pt):
    d = np.array([])
    for i in range(xs.shape[1]):
        d = np.append(d, np.linalg.norm(xs[:,i]-pt))
    return d

def hill(x,K,nH):
    return x**nH/(x**nH + K**nH)

# calculate distance to individual ghost states

g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

Gs = [g1,g2,g3,g4]
# M_SNT = np.zeros((len(Gs),len(sigma),nruns))

nth = 10
stateTCs = np.zeros((len(sigma),nruns,len(Gs)+1,int(timesteps/nth))) 

eps = 0.1 

for i in range(len(sigma)):
    print(i)
    for ii in range(nruns):
            stateTCs[i,ii,0,:] = simulations[i,ii,0,::nth]
            for iii in range(4):
                dist = distanceToPoint(simulations[i,ii,1:,::nth],Gs[iii])
                stateTCs[i,ii,iii+1,:] = dist
                # M_SNT[iii,i,ii] = stepsize*nth*len(dist[dist<eps])
                

#%% plot
myFig = plt.figure(figsize=(8.6*1.5*inCm,3*inCm))
n = 3 # select run

# noise level sigma: 5e-3
# ax1 = myFig.add_subplot(1,4,1)

# s = 6

# for i in range(4):
#     ax1.plot(stateTCs[s,n,0,:], hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)

# ax1.set_xlabel('time (a.u.)',fontsize=10)
# ax1.set_box_aspect(1/3)
# ax1.set_yticks([0,.5,1])
# ax1.set_ylim(0,1.1)
# ax1.set_xlim(0,2000)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.tight_layout()


sV = [0,3,6,9]

for j in range(4):

    # noise level sigma: 5e-2
    ax1 = myFig.add_subplot(1,4,1+j)
    
    s=sV[j]
    
    for i in range(4):
        ax1.plot(stateTCs[s,n,0,:], hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)
    
    
    ax1.set_title('$\\sigma=$'+"{:.4f}".format(sigma[s]),fontsize=8)
    ax1.set_xlabel('time (a.u.)',fontsize=10)
    ax1.set_box_aspect(1/3)
    ax1.set_yticks([0,.5,1])
    ax1.set_ylim(0,1.1)
    ax1.set_xlim(0,2000)
    if j == 3:  ax1.set_xlim(0,100)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

plt.subplots_adjust(top=0.873,
bottom=0.187,
left=0.076,
right=0.954,
hspace=0.2,
wspace=0.667)

# %% Figure 4 (g) - phase space trajectories colorcoded according to velocity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# noise levels sigma: 5e-3 and 5e-2
n = 3
sV = [0,3,6,9]
# sigs = [9,10]

# calculate velocities and percentiles

relV = [] # vector of relative velocities

p = 5 # percentile magnitude

p_l = [] # lower pth-percentiles
p_u = [] # upper pth-percentiles

for s in sV:

    simT,simX,simY,simZ = simulations[s,n,:,::10]
    
    vx=np.gradient(simX,simT)
    vy=np.gradient(simY,simT)
    vz=np.gradient(simZ,simT)
    v = np.sqrt(vx**2+vy**2+vz**2)
    relV.append(v)
    
    p_l.append(np.percentile(v, p))
    p_u.append(np.percentile(v, 100-p))

# set color scale boundaries, colormap and normalization

cmBounds = [min(p_l), max(p_u)]
cmap=cm.get_cmap('cool')
norm = plt.Normalize(cmBounds[0],cmBounds[1])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

# plot

myFig = plt.figure(figsize=(8.6*inCm*1.5,4*inCm))

for i in range(len(sV)):

    ax =  myFig.add_subplot(1,4,1+i,projection='3d')
    simT,simX,simY,simZ = simulations[sV[i],n,:,::10]
    
    points3D = np.array([simX, simY, simZ]).T.reshape(-1, 1, 3)
    segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)
    cols3D = relV[i]
    
    lc = Line3DCollection(segments3D, cmap='cool',norm=norm,lw=2)
    lc.set_array(cols3D)
    lc.set_linewidth(5)
    line = ax.add_collection3d(lc)
    
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

    

plt.subplots_adjust(top=0.873,
bottom=0.213,
left=0.066,
right=0.959,
hspace=0.065,
wspace=0.402)

#%% Figure 4 (h) - Period, time spent at ghosts, time spent switching between ghosts
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

valChecks = False # validity plots to check whether algorithm correctly identifies time spend in saddle vicinity

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
    print('progress:', "{:.0f}".format(100*(s/len(sigma))), '%')
    
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
            
            
            t_periods = []
            
            if len(np.where(thrCrossed == 1)[0]) > 0:
                t = np.min(np.where(thrCrossed == 1)[0]); seq = 0; seq_out = 0
            
                # Time of crossing
                
                while t < int(timesteps/nth)-1:
                    if thrCrossed[t] == 1:
                        if thrCrossed[t]-thrCrossed[t+1] == 0:
                            seq += 1
                        else:
                            periodsThreshold.append(seq)
                            seq = 0
                            t_periods.append(t)
                    else:
                        if thrCrossed[t]-thrCrossed[t+1] == 0:
                            seq_out += 1
                        else:
                            periodsThreshold_out.append(seq_out)
                            seq_out= 0
                            t_periods.append(t)
                    t+=1
            else:
                t_periods.append(0)
            
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
                    
    
#%% plot
    
myFig = plt.figure(figsize=(8.6*inCm*0.55,5*inCm))
   
plt.errorbar(sigma,avgPeriods_ghost,yerr=stdPeriods_ghost,color='r',capsize=1.5,fmt='-d',ms=3,label='period',lw=1)        
plt.errorbar(sigma,avgTimesAtGhosts,yerr=stdTimesAtGhosts,color='k',capsize=1.5,fmt='-o',ms=3,label='in ghost vicinity',lw=1)   
plt.errorbar(sigma,avgTimesNotAtGhosts,yerr=stdTimesNotAtGhosts,mfc='w',mec='k',ecolor='k',capsize=1.5,fmt=':sk',ms=3,label='not in ghost vicinity',lw=1)   

import math

for i in range(len(avgPeriods_ghost)):
    if math.isnan(avgPeriods_ghost[i]):
        plt.vlines(sigma[i],0,600,colors='m',linestyles='dotted',alpha=0.5)

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks(range(0,700,100),fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)
plt.tight_layout()
# plt.subplots_adjust(top=0.925, bottom=0.307, left=0.187, right=0.981, hspace=0.2, wspace=0.2)





