# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

This code reproduces the results shown in supplementary figure 4 from the study:
    
Koch D, Nandan A, Ramesan G, Tyukin I, Gorban A, Koseska A (2024): 
Ghost channels and ghost cycles guiding long transients in dynamical systems
In: Physical Review Letters (forthcoming)

IMPORTANT:
    The files "functions.py" and "models.py" need to be in the same folder as this script
"""

loadData = True

# Import packages etc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))

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

#%% attractor hopping
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Simulations metastable attractor hopping
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
t_end = 2000
stepsize = 0.01
timesteps = int(t_end/stepsize)
nruns = 5 #/30 number of repetitions
sigmaValues =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2] # noise levels

if loadData == False:
    # run and save simulations

    simulations = []
    for i in range(len(sigmaValues)): 
        sig = sigmaValues[i]
        print('Supplementarty figure 4: Simulations ' + str(int(i*100/len(sigmaValues))) + ' % complete.')
        ic = 0
        for n in range(nruns):
            ic += 1 
            if ic==4: ic = 0
            simDat = fun.RK4_na_noisy(models.sys_ghostCycle3D,[areas,steepness],ICs[ic],0,stepsize,t_end, sig, naFun = None,naFunParams = None)
            simulations.append(simDat)    
    
    simulations =  np.reshape(np.asarray(simulations),(len(sigmaValues),nruns,4,timesteps))
    np.save('data\\simdat_metastableAttractors.npy',simulations)

elif loadData == True:
    simulations = np.load('data\\simdat_metastableAttractors.npy') 
        
#%% Supplementary figure 4 (a) - Timecourses for selected noise levels  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# calculate distance to individual ghost states

g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

Gs = [g1,g2,g3,g4]

nth = 10
stateTCs = np.zeros((len(sigmaValues),nruns,len(Gs)+1,int(timesteps/nth))) 

eps = 0.1 

for i in range(len(sigmaValues)):
    print('Supplementary figure 4 (a): data analysis ' + str(int(i*100/len(sigmaValues))) + ' % complete.')
    for ii in range(nruns):
            stateTCs[i,ii,0,:] = simulations[i,ii,0,::nth]
            for iii in range(4):
                dist = fun.distanceToPoint(simulations[i,ii,1:,::nth],Gs[iii])
                stateTCs[i,ii,iii+1,:] = dist
                

# plot
myFig = plt.figure(figsize=(8.6*1.5*inCm,3*inCm))
n = 3 # select run

sV = [0,3,6,9]

for j in range(4):

    # noise level sigma: 5e-2
    ax1 = myFig.add_subplot(1,4,1+j)
    
    s=sV[j]
    
    for i in range(4):
        ax1.plot(stateTCs[s,n,0,:], fun.hill(stateTCs[s,n,i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=1) #cm.get_cmap('magma',5)(i)
    
    
    ax1.set_title('$\\sigma=$'+"{:.4f}".format(sigmaValues[s]),fontsize=8)
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

# phase space trajectories colorcoded according to velocity

# noise levels sigma: 5e-3 and 5e-2
n = 3
sV = [0,3,6,9]

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


print('Suplementary Figure 4 (a): plotting complete.')

#%% Supplementary figure 4 (b) - Period, time spent at metastable attractor states, time spent switching between metastable attractor states
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

valChecks = False # validity plots to check whether algorithm correctly identifies time spend in state vicinity

g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])

Gs = [g1,g2,g3,g4]

eps = 0.1 
nth = 10

allPeriods_ghost = []
allTimesAtMetaState = []
allTimesNotAtMetaState = []

avgPeriods_ghost = []
avgTimesAtMetaState = []
avgTimesNotAtMetaState = []

stdPeriods_ghost = []
stdTimesAtMetaState = []
stdTimesNotAtMetaState = []

for s in range(len(sigmaValues)):
    print('Supplementary figure 4 (b): data analysis ' + str(int(s*100/len(sigmaValues))) + ' % complete.')
    
    #individual values all runs
    allPeriods_rs = []
    allTimesAtMetaState_rs = []
    allTimesNotAtMetaState_rs = []

    #Avg all runs
    avgPeriods_rs = []
    avgTimesAtMetaState_rs = []
    avgTimesNotAtMetaState_rs = []
    
    #SD all runs
    stdPeriods_rs = []
    stdTimesAtMetaState_rs = []
    stdTimesNotAtMetaState_rs = []
    
    
    for n in range(nruns):
        
            sim_nth = simulations[s,n,:,::nth]
        
            # Thresholding (based on distances)
            thrCrossed = []
            for t in range(int(timesteps/nth)):
                if fun.hill(stateTCs[s,n,1,t],0.3,-3) < 0.25:
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
            timesAtMetaState_run = []
            timesNotAtMetaState_run = []
            
            tts = []
            
            if all([s == 0, n == 0, valChecks == True]):
                plt.figure()
                
                for g in range(4):
                  plt.plot(stateTCs[s,n,0,:], fun.hill(stateTCs[s,n,g+1,:],0.3,-3) ,'-', label='G'+str(g+1), color = tcColors[g], lw=3,alpha=0.5) 
                  
                for t in range(0,2*nrFullPeriods-2,2):
                    plt.vlines(t_periods[t]*nth*stepsize,0,1,'k',lw=3)
                plt.xlim(0,300)
            
            for t in range(0,2*nrFullPeriods-2,2):
                
                t_AtMetaState = 0
                
                for g in range(4):
                    for tt in range(t_periods[t],t_periods[t+2],nth_dist):
                        dist = np.linalg.norm(sim_nth[1:,tt]-Gs[g])
                        
                        if dist<eps:
                            t_AtMetaState += 1*nth_dist
                            
                        if all([s == 0, n == 0, valChecks == True]):
                            if dist<eps:
                                plt.scatter(tt*nth*stepsize,1.1,c=tcColors[g],alpha=1)
                                
                timesAtMetaState_run.append(t_AtMetaState*nth*stepsize)
                timesNotAtMetaState_run.append((t_periods[t+2]-t_periods[t])*stepsize*nth-t_AtMetaState*nth*stepsize)
                periods_run.append((t_periods[t+2]-t_periods[t])*stepsize*nth)              
                    
            allPeriods_rs.append(np.asarray(periods_run))
            allTimesAtMetaState_rs.append(np.asarray(timesAtMetaState_run))
            allTimesNotAtMetaState_rs.append(np.asarray(timesNotAtMetaState_run))
             
            avgPeriods_rs.append(np.mean(np.asarray(periods_run)))
            avgTimesAtMetaState_rs.append(np.mean(np.asarray(timesAtMetaState_run)))
            avgTimesNotAtMetaState_rs.append(np.mean(np.asarray(timesNotAtMetaState_run)))
            
            stdPeriods_rs.append(np.std(np.asarray(periods_run)))
            stdTimesAtMetaState_rs.append(np.std(np.asarray(timesAtMetaState_run)))
            stdTimesNotAtMetaState_rs.append(np.std(np.asarray(timesNotAtMetaState_run)))
            
    
    allPeriods_ghost.append(allPeriods_rs)
    allTimesAtMetaState.append(allTimesAtMetaState_rs)
    allTimesNotAtMetaState.append(allTimesNotAtMetaState_rs)

    avgPeriods_ghost.append(np.mean(np.asarray(avgPeriods_rs)))
    avgTimesAtMetaState.append(np.mean(np.asarray(avgTimesAtMetaState_rs)))
    avgTimesNotAtMetaState.append(np.mean(np.asarray(avgTimesNotAtMetaState_rs)))

    stdPeriods_ghost.append( (np.mean(np.asarray(stdPeriods_rs)**2))**0.5 )
    stdTimesAtMetaState.append( (np.mean(np.asarray(stdTimesAtMetaState_rs)**2))**0.5 )
    stdTimesNotAtMetaState.append( (np.mean(np.asarray(stdTimesNotAtMetaState_rs)**2))**0.5 )
                    
    
# plot
    
myFig = plt.figure(figsize=(8.6*inCm*0.55,5*inCm))
   
plt.errorbar(sigmaValues,avgPeriods_ghost,yerr=stdPeriods_ghost,color='r',capsize=1.5,fmt='-d',ms=3,label='period',lw=1)        
plt.errorbar(sigmaValues,avgTimesAtMetaState,yerr=stdTimesAtMetaState,color='k',capsize=1.5,fmt='-o',ms=3,label='in ghost vicinity',lw=1)   
plt.errorbar(sigmaValues,avgTimesNotAtMetaState,yerr=stdTimesNotAtMetaState,mfc='w',mec='k',ecolor='k',capsize=1.5,fmt=':sk',ms=3,label='not in ghost vicinity',lw=1)   

import math

for i in range(len(avgPeriods_ghost)):
    if math.isnan(avgPeriods_ghost[i]):
        plt.vlines(sigmaValues[i],0,600,colors='m',linestyles='dotted',alpha=0.5)

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks(range(0,700,100),fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)
plt.tight_layout()

print('Suplementary Figure 4 (b): plotting complete.')

