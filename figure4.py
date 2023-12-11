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

loadData = True

# Import packages etc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import functions as fun 
import models
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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


#%% Heteroclinic cycle
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Simulations heteroclinic cycle 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

# set random seed (optional)
seed_int = 1
np.random.seed(seed_int)

# model parameters
alpha = np.ones(3)*2
beta = np.ones(3)
v = np.ones(3)*4
par_Horchler = [alpha, beta, v]

# simulation settings
ICs = [[1,0,0],[0,1,0],[0,0,1]] # initial conditions
t_end = 1000
stepsize = 0.01
timesteps = int(t_end/stepsize)
nruns = 30 # number of repetitions
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2] # noise levels


if loadData == False:
    # run and save simulations
    simulations = []
    for s in sigma:
        print(s)
        ic = 0
        for n in range(nruns):
            ic += 1 
            if ic==3: ic = 0
            simDat = fun.RK4_na_noisy_pos(models.Horchler2015,par_Horchler,ICs[ic],0,stepsize,t_end, s, naFun = None,naFunParams = None)    
            simulations.append(simDat)
    
    simulations =  np.reshape(np.asarray(simulations),(len(sigma),nruns,4,timesteps))
    np.save('simdat_Horchler2015_final.npy',simulations)
    
elif loadData == True:
    simulations = np.load('simdat_Horchler2015_final.npy')

#%% Figure 4 (b) - Timecourses for selected noise levels  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
myFig = plt.figure(figsize=(8.6*inCm/2,6*inCm))

# noise level sigma: 5e-3

ax1 = myFig.add_subplot(2,1,1)
simT,simX,simY,simZ = simulations[5,0,:,:]

ax1.plot(simT,simX ,'-', color = tcColors[0], label='$a_{1}$',lw=1)
ax1.plot(simT,simY ,'-', color = tcColors[1],label='$a_{2}$', lw=1)
ax1.plot(simT,simZ ,'-', color = tcColors[2], label='$a_{3}$',lw=1)

ax1.set_xlabel('time (a.u.)',fontsize=10)
ax1.set_box_aspect(1/3)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


# noise level sigma: 5e-2

ax2 = myFig.add_subplot(2,1,2)

simT,simX,simY,simZ = simulations[8,0,:,:]

ax2.plot(simT,simX ,'-', color = tcColors[0], label='$a_{1}$',lw=1)
ax2.plot(simT,simY ,'-', color = tcColors[1], label='$a_{2}$', lw=1)
ax2.plot(simT,simZ ,'-', color = tcColors[2], label='$a_{3}$',lw=1)

ax2.set_xlabel('time (a.u.)',fontsize=10)
ax2.set_box_aspect(1/3)
ax2.set_yticks([0,.5,1])
ax2.set_ylim(0,1.1)
ax2.set_xlim(0,100)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplots_adjust(top=0.936, bottom=0.154, left=0.163, right=0.942, hspace=0.0, wspace=0.2)

#%% Figure 4 (c) - phase space trajectories colorcoded according to velocity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# noise levels sigma: 5e-3 and 5e-2

sigs = [5,8]

# calculate velocities and percentiles

relV = [] # vector of relative velocities

p = 5 # percentile magnitude

p_l = [] # lower pth-percentiles
p_u = [] # upper pth-percentiles

for s in sigs:

    simT,simX,simY,simZ = simulations[s,0,:,::10]
    
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

myFig = plt.figure(figsize=(8.6*inCm,4*inCm))

for i in range(len(sigs)):

    ax =  myFig.add_subplot(1,2,1+i,projection='3d')
    
    simT,simX,simY,simZ = simulations[sigs[i],0,:,::10]
    
    points3D = np.array([simX, simY, simZ]).T.reshape(-1, 1, 3)
    segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)
    cols3D = relV[i]
    
    lc = Line3DCollection(segments3D, cmap='cool',norm=norm,lw=2)
    lc.set_array(cols3D)
    lc.set_linewidth(5)
    line = ax.add_collection3d(lc)
    
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



#%% calculate time spend at saddles

# SN1 = np.array([1,0,0])
# SN2 = np.array([0,1,0])
# SN3 = np.array([0,0,1])

# SNs = [SN1, SN2, SN3]
# M_SNT = np.zeros((len(SNs),len(sigma),nruns))
# eps = 0.1 


# for i in range(len(sigma)):
#     print(i)
#     for ii in range(nruns):
#             nth = 20
#             for iii in range(3):
#                 # print(i,ii,iii)
#                 # dist = fun.euklDist_TvP(simulations[iii,i,ii,::nth].T, SNs[iii])
#                 dist = fun.distanceToPoint(simulations[i,ii,1:,::nth],SNs[iii])
#                 M_SNT[iii,i,ii] = stepsize*nth*len(dist[dist<eps])
                
#%% Figure 4 (d) - Period, time spent at saddles, time spent switching between saddles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

valChecks = False # validity plots to check whether algorithm correctly identifies time spend in saddle vicinity

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
    print('progress:', "{:.0f}".format(100*(s/len(sigma))), '%')
    
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
                    
    
# plot
myFig = plt.figure(figsize=(8.6*inCm/2,4*inCm))

plt.errorbar(sigma,avgPeriods_saddle,yerr=stdPeriods_saddle,color='r',capsize=1.5,fmt='-d',ms=3,label='period',lw=1)      
plt.errorbar(sigma,avgTimesAtSaddles,yerr=stdTimesAtSaddles,color='k',capsize=1.5,fmt='-o',ms=3,label='in saddle vicinity period',lw=1)   
plt.errorbar(sigma,avgTimesNotAtSaddles,yerr=stdTimesNotAtSaddles,mfc='w',mec='k',ecolor='k',capsize=1.5,fmt=':sk',ms=3,label='not in saddle vicinity',lw=1)   

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)

plt.subplots_adjust(top=0.925, bottom=0.307, left=0.187, right=0.981, hspace=0.2, wspace=0.2)



#%% Ghost cycle
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Simulations ghost cycle 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

# set random seed (optional)
seed_int = 1
np.random.seed(seed_int)

# model parameters
areas = [[0,1,0,1,0,1],[1,2,0,1,0,1],[1,2,1,2,0,1],[0,1,1,2,0,1]]
steepness = 10

# simulation settings
ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5]] # initial conditions
t_end = 1000
stepsize = 0.01
nruns = 30 # number of repetitions
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
            simDat = fun.RK4_na_noisy(models.sys_ghostCycle3D,[areas,steepness],ICs[ic],0,stepsize,t_end, s, naFun = None,naFunParams = None)
            simulations.append(simDat)    
    
    simulations =  np.reshape(np.asarray(simulations),(len(sigma),nruns,4,timesteps))
    np.save('simdat_Ghostcycle.npy',simulations)

elif loadData == True:
    simulations = np.load('simdat_Ghostcycle.npy') 


        
#%% Figure 4 (f) - Timecourses for selected noise levels  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                dist = fun.distanceToPoint(simulations[i,ii,1:,::nth],Gs[iii])
                stateTCs[i,ii,iii+1,:] = dist
                # M_SNT[iii,i,ii] = stepsize*nth*len(dist[dist<eps])
                

# plot
myFig = plt.figure(figsize=(8.6*inCm/2,6*inCm))
n = 5 # select run

# noise level sigma: 5e-3
ax1 = myFig.add_subplot(2,1,1)

s = 0

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], fun.hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=10)
ax1.set_box_aspect(1/3)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()


# noise level sigma: 5e-2
ax1 = myFig.add_subplot(2,1,2)
s = 8

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], fun.hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)


ax1.set_xlabel('time (a.u.)',fontsize=10)
ax1.set_box_aspect(1/3)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,100)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplots_adjust(top=0.936, bottom=0.154, left=0.163, right=0.942, hspace=0.0, wspace=0.2)

#%% Figure 4 (g) - phase space trajectories colorcoded according to velocity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# noise levels sigma: 5e-3 and 5e-2

sigs = [5,8]

# calculate velocities and percentiles

relV = [] # vector of relative velocities

p = 5 # percentile magnitude

p_l = [] # lower pth-percentiles
p_u = [] # upper pth-percentiles

for s in sigs:

    simT,simX,simY,simZ = simulations[s,0,:,::10]
    
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

myFig = plt.figure(figsize=(8.6*inCm,4*inCm))

for i in range(len(sigs)):

    ax =  myFig.add_subplot(1,2,1+i,projection='3d')
    simT,simX,simY,simZ = simulations[sigs[i],0,:,::10]
    
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
    plt.tight_layout()

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
                if fun.hill(stateTCs[s,n,1,t],0.3,-3) < 0.25:
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
                  plt.plot(stateTCs[s,n,0,:], fun.hill(stateTCs[s,n,g+1,:],0.3,-3) ,'-', label='G'+str(g+1), color = tcColors[g], lw=3,alpha=0.5) 
                  
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
                    
    
# plot
    
myFig = plt.figure(figsize=(8.6*inCm/2,4*inCm))
   
plt.errorbar(sigma,avgPeriods_ghost,yerr=stdPeriods_ghost,color='r',capsize=1.5,fmt='-d',ms=3,label='period',lw=1)        
plt.errorbar(sigma,avgTimesAtGhosts,yerr=stdTimesAtGhosts,color='k',capsize=1.5,fmt='-o',ms=3,label='in ghost vicinity',lw=1)   
plt.errorbar(sigma,avgTimesNotAtGhosts,yerr=stdTimesNotAtGhosts,mfc='w',mec='k',ecolor='k',capsize=1.5,fmt=':sk',ms=3,label='not in ghost vicinity',lw=1)   

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks([0,25,50,75],fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)

plt.subplots_adjust(top=0.925, bottom=0.307, left=0.187, right=0.981, hspace=0.2, wspace=0.2)

#%% Figure S3 (d) 

# set random seed (optional)
seed_int = 3
np.random.seed(seed_int)


alpha = np.ones(3)*2
beta = np.ones(3)
v = np.ones(3)*4
par_Horchler = [alpha, beta, v]
stepsize = 0.01
t_end = 6000   #
timesteps = int(t_end/stepsize)

s = 0.0001 # sigma

simDatGhost = fun.RK4_na_noisy(models.sys_ghostCycle3D,[areas,steepness],[0.5,0.5,0.5],0,stepsize,t_end, s, naFun = None,naFunParams = None)
simDatHorchler = fun.RK4_na_noisy_pos(models.Horchler2015,par_Horchler,[1,0,0],0,stepsize,t_end, s, naFun = None,naFunParams = None)    

nth = 10

# positions of saddles
SN1 = np.array([1,0,0])
SN2 = np.array([0,1,0])
SN3 = np.array([0,0,1])

# positions of ghosts
g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

SNs = [SN1, SN2, SN3]
Gs = [g1,g2,g3,g4]

timeAtGhosts = []
timeAtSNs = []

eps = 0.1 # size of epsilon vicinity

for n in range(4):
    
    # for each time point determine if system is in epsilon vicinity of saddle/ghost point
    
    distGhost = fun.distanceToPoint(simDatGhost[1:,::nth],Gs[n])
    if n < 3: distHC = fun.distanceToPoint(simDatHorchler[1:,::nth],SNs[n])
    
    bGhost = []
    if n < 3: bHC = []

    for i in range(int(timesteps/nth)):
        if distGhost[i] < eps:
            bGhost.append(1)
        else:
            bGhost.append(0)
        if n < 3:    
            if distHC[i] < eps:
                bHC.append(1)
            else:
                bHC.append(0)
    bGhost = np.asarray(bGhost)
    if n < 3: bHC = np.asarray(bHC)
    
    
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
            if bHC[t] == 1:
                if bHC[t]-bHC[t+1] == 0:
                    seq += 1
                else:
                    timeAtSNs.append(seq*nth*stepsize)
                    seq = 0
            t+=1
                
#%% plot
myFig = plt.figure(figsize=(14*inCm/2,6*inCm))
plt.scatter([np.mean(timeAtSNs[1:])], [400], marker='v', s =100, color='green', label='saddles',alpha=0.5)
plt.scatter([np.mean(timeAtGhosts[1:])], [400],marker='v', s =100, color='blue', label='ghosts',alpha=0.5)
plt.hist(timeAtSNs[1:],alpha=0.5, range=(9,15),bins=30, color='green')
plt.hist(timeAtGhosts[1:],alpha=0.5,range=(9,15),bins=30, color='blue')


plt.subplots_adjust(top=0.936, bottom=0.261, left=0.235, right=0.945, hspace=0.2, wspace=0.2)

plt.xlabel('time spent within $\\epsilon$-vicinity (a.u.)')
plt.ylabel('count')
plt.legend()

print('Difference between time at ghosts and time at saddle:' + str(np.abs(np.mean(timeAtSNs[1:]) - np.mean(timeAtGhosts[1:]))) + ' a.u. ' + '; in percent:'  + str(100 + 100*( - np.mean(timeAtSNs[1:]))/np.mean(timeAtGhosts[1:])))

#%% Figure S3 (a)-(c)

simulations = np.load('simdat_Ghostcycle.npy')

myFig = plt.figure(figsize=(12,3))

ax1 = myFig.add_subplot(1,3,1)

s = 0; n = 0

# raw timecourses 

ax1.plot(simulations[s,n,0,:], simulations[s,n,3,:] ,'-', label='z', color = 'mediumblue', lw=3)
ax1.plot(simulations[s,n,0,:], simulations[s,n,1,:] ,'-', label='x', color = 'k', lw=3)
ax1.plot(simulations[s,n,0,:], simulations[s,n,2,:] ,'-', label='y', color = 'crimson', lw=3)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$\overline{x}(t) = ( x(t), y(t), z(t) )^T$',fontsize=18)
ax1.set_ylim(0.4,1.6)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()


# Euclidean distance to ghosts

ax1 = myFig.add_subplot(1,3,2)

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], stateTCs[s,n,i+1,:] ,'-', label='G'+str(i+1), color = tcColors[i], lw=3) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$|| \overline{x}(t) - G_i ||$',fontsize=18)
ax1.set_ylim(0,1.5)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Hill function applied to Euclidean distance

ax1 = myFig.add_subplot(1,3,3)

for i in range(4):
    ax1.plot(stateTCs[s,n,0,:], fun.hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=3) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$\Theta(|| \overline{x}(t) - G_i ||)$',fontsize=18)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplots_adjust(top=0.922, bottom=0.259, left=0.068, right=0.988, hspace=0.2, wspace=0.285)



