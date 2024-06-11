# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

This code reproduces the results shown in supplementary figure 3 from the study:
    
Koch D, Nandan A, Ramesan G, Tyukin I, Gorban A, Koseska A (2024): 
Ghost channels and ghost cycles guiding long transients in dynamical systems
In: Physical Review Letters (forthcoming)

"""

# Import packages etc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
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
tcColors = ['royalblue','tomato','mediumaquamarine','mediumorchid']

#%% Suppelementary figure 3 (a)-(c)
seed_int = 1
np.random.seed(seed_int)

# model parameters
areas = [[0,1,0,1,0,1],[1,2,0,1,0,1],[1,2,1,2,0,1],[0,1,1,2,0,1]]
steepness = 10

# simulation settings
t_end = 1000
stepsize = 0.01
timesteps = int(t_end/stepsize)

# simulation
simDat = fun.RK4_na_noisy(models.sys_ghostCycle3D,[areas,steepness],[0.5,1.5,0.5],0,stepsize,t_end, 1e-4, naFun = None,naFunParams = None)

# positions of ghosts in phase space
g1 = np.array([0.5,0.5,0.5])
g2 = np.array([0.5,1.5,0.5])
g3 = np.array([1.5,0.5,0.5])
g4 = np.array([1.5,1.5,0.5])
Gs = [g1,g2,g3,g4]

# settings for euclidean distance calculations
nth = 10
stateTCs = np.zeros((len(Gs)+1,int(timesteps/nth))) 
eps = 0.1 

stateTCs[0,:] = simDat[0,::nth]
for iii in range(4):
    dist = fun.distanceToPoint(simDat[1:,::nth],Gs[iii])
    stateTCs[iii+1,:] = dist     

myFig = plt.figure(figsize=(12,3))

ax1 = myFig.add_subplot(1,3,1)

# raw timecourses 

ax1.plot(simDat[0,:], simDat[3,:] ,'-', label='z', color = 'mediumblue', lw=3)
ax1.plot(simDat[0,:], simDat[1,:] ,'-', label='x', color = 'k', lw=3)
ax1.plot(simDat[0,:], simDat[2,:] ,'-', label='y', color = 'crimson', lw=3)

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
    ax1.plot(stateTCs[0,:], stateTCs[i+1,:] ,'-', label='G'+str(i+1), color = tcColors[i], lw=3) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$|| \overline{x}(t) - G_i ||$',fontsize=18)
ax1.set_ylim(0,1.5)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Hill function applied to Euclidean distance
ax1 = myFig.add_subplot(1,3,3)

for i in range(4):
    ax1.plot(stateTCs[0,:], fun.hill(stateTCs[i+1,:],0.3,-3) ,'-', label='G'+str(i+1), color = tcColors[i], lw=3) #cm.get_cmap('magma',5)(i)

ax1.set_xlabel('time (a.u.)',fontsize=18)
ax1.set_ylabel('$\Theta(|| \overline{x}(t) - G_i ||)$',fontsize=18)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(0,1.1)
ax1.set_xlim(0,150)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplots_adjust(top=0.922, bottom=0.259, left=0.068, right=0.988, hspace=0.2, wspace=0.285)


#%% Suppelementary figure 3 (d)

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
                
# plot
myFig = plt.figure(figsize=(14*inCm/2,6*inCm))
plt.scatter([np.mean(timeAtSNs[1:])], [400], marker='v', s =100, color='green', label='saddles',alpha=0.5)
plt.scatter([np.mean(timeAtGhosts[1:])], [400],marker='v', s =100, color='blue', label='ghosts',alpha=0.5)
plt.hist(timeAtSNs[1:],alpha=0.5, range=(9,15),bins=30, color='green')
plt.hist(timeAtGhosts[1:],alpha=0.5,range=(9,15),bins=30, color='blue')


plt.subplots_adjust(top=0.936, bottom=0.261, left=0.235, right=0.945, hspace=0.2, wspace=0.2)

plt.xlabel('time spent within $\\epsilon$-vicinity (a.u.)')
plt.ylabel('count')
plt.legend()

print('Difference between time at ghosts and time at saddle: ' + " {:.3f}".format(np.abs(np.mean(timeAtSNs[1:]) - np.mean(timeAtGhosts[1:]))) + ' a.u. ' + '; in percent:'  + " {:.3f}".format(100 + 100*( - np.mean(timeAtSNs[1:]))/np.mean(timeAtGhosts[1:])) + ' %')

