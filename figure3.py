# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:43:08 2023

@author: Daniel Koch
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import functions as fun 
import models
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))

# some plotting settings
plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
inCm = 1/2.54

#%% ~~~~~~~~~~~~~~~~~~~ Heteroclinic channel ~~~~~~~~~~~~~~~~~~~
t_end = 100     #
dt = 0.05
timesteps = int(t_end/dt)
timepoints = np.linspace(0, t_end, timesteps+1)

x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

#%% figure 3 (a) - simulation and saving data

seeds = [2,5]

fileNames = ['data\\simdat_HC4_6ICs_s5e-3.npy','data\\simdat_HC4_6ICs_s5e-2.npy']

sigmaVals = [5e-3,5e-2]

print('Figure 3 (a): Simulating...')
for i in range(2):   
    
    np.random.seed(seeds[i])
    
    simDat = []
    
    for ic in ICs:
        sim = fun.RK4_na_noisy(models.sys_HC4,[],ic,0,dt,t_end, sigmaVals[i], naFun = None,naFunParams = None)
        simDat.append(sim)
            
    simDat = np.asarray(simDat)
    simDat = np.reshape(simDat, (6,3,2000))
    
    np.save(fileNames[i], simDat)
print('Figure 3 (a): Simulation complete.')

#%% figure 3 (a) - loading data and plotting

simDat_loNoise = np.load('data\\simdat_HC4_6ICs_s5e-3.npy')
simDat_hiNoise = np.load('data\\simdat_HC4_6ICs_s5e-2.npy')

norm = plt.Normalize(0,8)
cmap=cm.get_cmap('RdPu_r')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

X = np.linspace(-0.1,5, 20)
Y = np.linspace(-0.1,5, 20)
grid = np.meshgrid(X, Y)

fig, ax = plt.subplots(figsize=(5,5))
fun.plot_streamline(ax,models.sys_HC4,[] ,10, grid, lw = 1.5)
plt.title('4-saddle heteroclinic channel, $\sigma=$'+"{:.4f}".format(5e-3),fontsize=16)
plt.xlabel('x',fontsize=16);plt.ylabel('y',fontsize=16)
plt.xlim(-0.35,5)
plt.ylim(-0.35,5)


for ic in ICs:
    idx = np.where(ICs == ic)[0][0]
    col = np.asarray(cmap(norm(idx))[0:3])
    ax.plot(simDat_loNoise[idx,1,:],simDat_loNoise[idx,2,:],color=col,lw=2.5)
    ax.scatter(simDat_loNoise[idx,1,0],simDat_loNoise[idx,2,0],marker='o',s=60,edgecolors='black',color=col)
    
x1,y1 = -0.2,1
x2,y2 = 1,-0.2
x3,y3 = np.asarray([x1,y1])+3.5
x4,y4 = np.asarray([x2,y2])+3.5

ax.plot([x1,x2,x4,x3,x1],np.asarray([y1,y2,y4,y3,y1])-0.1,'-k',lw=2)

    
fig, ax = plt.subplots(figsize=(5,5))
fun.plot_streamline(ax,models.sys_HC4,[] ,10, grid, lw = 1.5)
plt.title('4-saddle heteroclinic channel, $\sigma=$'+"{:.4f}".format(5e-2),fontsize=16)
plt.xlabel('x',fontsize=16);plt.ylabel('y',fontsize=16)
plt.xlim(-0.35,5)
plt.ylim(-0.35,5)


for ic in ICs:
    idx = np.where(ICs == ic)[0][0]
    col = np.asarray(cmap(norm(idx))[0:3])
    ax.plot(simDat_hiNoise[idx,1,:],simDat_hiNoise[idx,2,:],color=col,lw=2.5)
    ax.scatter(simDat_hiNoise[idx,1,0],simDat_hiNoise[idx,2,0],marker='o',s=60,edgecolors='black',color=col)

x1,y1 = -0.2,1
x2,y2 = 1,-0.2
x3,y3 = np.asarray([x1,y1])+3.5
x4,y4 = np.asarray([x2,y2])+3.5

ax.plot([x1,x2,x4,x3,x1],np.asarray([y1,y2,y4,y3,y1])-0.1,'-k',lw=2)

#%% ~~~~~~~~~~~~~~~~~~~ Ghost channel ~~~~~~~~~~~~~~~~~~~

dt = 0.05
t_end = 300     #
timesteps = int(t_end/dt)
timepoints = np.linspace(0, t_end, timesteps+1)

x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

#%% figure 3 (b) - simulation and saving data

seeds = [1,1]

fileNames = ['data\\simdat_GC4_6ICs_s5e-3.npy','data\\simdat_GC4_6ICs_s5e-2.npy']

sigmaVals = [5e-3,5e-2]

print('Figure 3 (b): Simulating...')
for i in range(2):   
    
    np.random.seed(seeds[i])
    
    simDat = []
    
    for ic in ICs:
        sim = fun.RK4_na_noisy(models.sys_ghost4,[],ic,0,dt,t_end, sigmaVals[i], naFun = None,naFunParams = None)
        simDat.append(sim)
        ax.plot(sim[1,:],sim[2,:],lw=2.5,color=col)
        
    simDat = np.asarray(simDat)
    simDat = np.reshape(simDat, (6,3,6000))
    np.save(fileNames[i], simDat)
print('Figure 3 (b): Simulation complete.')

#%% figure 3 (b) - loading data and plotting

simDat_loNoise = np.load('data\\simdat_GC4_6ICs_s5e-3.npy')
simDat_hiNoise = np.load('data\\simdat_GC4_6ICs_s5e-2.npy')

norm = plt.Normalize(0,8)
cmap=cm.get_cmap('RdPu_r')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

X = np.linspace(-0.1,5, 20)
Y = np.linspace(-0.1,5, 20)
grid = np.meshgrid(X, Y)

fig, ax = plt.subplots(figsize=(5,5))
fun.plot_streamline(ax,models.sys_ghost4,[] ,10, grid, lw = 1.5)
plt.title('4-ghost channel, $\sigma=$'+"{:.4f}".format(sigmaVals[0]),fontsize=16)
plt.xlabel('x',fontsize=16);plt.ylabel('y',fontsize=16)
plt.xlim(-0.35,5)
plt.ylim(-0.35,5)

for ic in ICs:
    idx = np.where(ICs == ic)[0][0]
    col = np.asarray(cmap(norm(idx))[0:3])
    ax.plot(simDat_loNoise[idx,1,:],simDat_loNoise[idx,2,:],color=col,lw=2.5)
    ax.scatter(simDat_loNoise[idx,1,0],simDat_loNoise[idx,2,0],marker='o',s=60,edgecolors='black',color=col)
    
x1,y1 = -0.2,1
x2,y2 = 1,-0.2
x3,y3 = np.asarray([x1,y1])+3.5
x4,y4 = np.asarray([x2,y2])+3.5

ax.plot([x1,x2,x4,x3,x1],np.asarray([y1,y2,y4,y3,y1])-0.1,'-k',lw=2)

    
fig, ax = plt.subplots(figsize=(5,5))
fun.plot_streamline(ax,models.sys_ghost4,[] ,10, grid, lw = 1.5)
plt.title('4-ghost channel, $\sigma=$'+"{:.4f}".format(sigmaVals[1]),fontsize=16)
plt.xlabel('x',fontsize=16);plt.ylabel('y',fontsize=16)
plt.xlim(-0.35,5)
plt.ylim(-0.35,5)

for ic in ICs:
    idx = np.where(ICs == ic)[0][0]
    col = np.asarray(cmap(norm(idx))[0:3])
    ax.plot(simDat_hiNoise[idx,1,:],simDat_hiNoise[idx,2,:],color=col,lw=2.5)
    ax.scatter(simDat_hiNoise[idx,1,0],simDat_hiNoise[idx,2,0],marker='o',s=60,edgecolors='black',color=col)

x1,y1 = -0.2,1
x2,y2 = 1,-0.2
x3,y3 = np.asarray([x1,y1])+3.5
x4,y4 = np.asarray([x2,y2])+3.5

ax.plot([x1,x2,x4,x3,x1],np.asarray([y1,y2,y4,y3,y1])-0.1,'-k',lw=2)



#%% ~~~~~~~~~~~~~~~~~~~ Trajectory quantifications for heteroclinic and Ghost channels ~~~~~~~~~~~~~~~~~~~
# caution: running these simulations can take multiple hours.

#%% figure 3 (c) - loading data and plotting - simulation and saving data

dt = 0.05
t_end = 500
timesteps = int(t_end/dt)
timepoints = np.linspace(0, t_end, timesteps+1)

x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

sigmaValues =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]

nruns = 30 # only take even integers

# Heteroclinic channels
simulations_HC = []
np.random.seed(1)
for i in range(len(sigmaValues)): 
    sig = sigmaValues[i]
    print('Figure 3 (c): Simulations for heteroclinic channel ' + str(int(i*100/len(sigmaValues))) + ' % complete.')
    for ic in ICs:
        for n in range(nruns):
            sim = fun.RK4_na_noisy(models.sys_HC4,[],ic,0,dt,t_end, sig, naFun = None,naFunParams = None)
            simulations_HC.append(sim)
       
simulations_HC =  np.reshape(np.asarray(simulations_HC),(1,len(sigmaValues),len(ICs),nruns,3,timesteps))
np.save('simdat_HC4_noisy_30runs.npy',simulations_HC)

# Ghost channels

simulations_ghost = []
np.random.seed(1)
for i in range(len(sigmaValues)): 
    sig = sigmaValues[i]
    print('Figure 3 (c): Simulations for ghost channel ' + str(int(i*100/len(sigmaValues))) + ' % complete.')
    for ic in ICs:
        for i in range(nruns):
            sim = fun.RK4_na_noisy(models.sys_ghost4,[],ic,0,dt,t_end, sig, naFun = None,naFunParams = None)
            simulations_ghost.append(sim)
 
simulations_ghost =  np.reshape(np.asarray(simulations_ghost),(len(sigmaValues),len(ICs),nruns,3,timesteps))
np.save('simdat_GC4_noisy_30runs.npy',simulations_ghost)

#%% figure 3 (c) - loading and analyzing data

simulationsHC = np.load('simdat_HC4_noisy_30runs.npy')
simulationsHC_ = np.reshape(simulationsHC, (simulationsHC.shape[0],simulationsHC.shape[1],simulationsHC.shape[2]*simulationsHC.shape[3],simulationsHC.shape[4],simulationsHC.shape[5]))
simulationsHC_= simulationsHC_[0]

simulationsGC = np.load('simdat_GC4_noisy_30runs.npy')
simulationsGC_ = np.reshape(simulationsGC, (simulationsGC.shape[0],simulationsGC.shape[1]*simulationsGC.shape[2],simulationsGC.shape[3],simulationsGC.shape[4]))

sigmaValues =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
nruns = 30 

EDsHC = []
SDsHC = []
EDsGC = []
SDsGC = []
 
for i in range(len(sigmaValues)):       
    print('Figure 3 (c): data analysis ' + str(int(i*100/len(sigmaValues))) + ' % complete.')
    
    length_ICs = 6
    
    s1 = simulationsHC_[i,:int(nruns*length_ICs/2),1:,::20]
    s1_ = np.swapaxes(s1, 1, 2)
    s2 = simulationsHC_[i,int(nruns*length_ICs/2):,1:,::20]
    s2_ = np.swapaxes(s2, 1, 2)
    
    best_path = fun.dtw_getWarpingPaths(s1_,s2_,'multiple repetitions', 'True')
    ED, SD = fun.euklDist_trajectory(s1_[:,best_path[0],:],s2_[:,best_path[1],:],'replicate')
    EDsHC.append(ED)
    SDsHC.append(SD)
    
    s1 = simulationsGC_[i,:int(nruns*length_ICs/2),1:,::20]
    s1_ = np.swapaxes(s1, 1, 2)
    s2 = simulationsGC_[i,int(nruns*length_ICs/2):,1:,::20]
    s2_ = np.swapaxes(s2, 1, 2)
    
    best_path = fun.dtw_getWarpingPaths(s1_,s2_,'multiple repetitions', 'True')
    ED, SD = fun.euklDist_trajectory(s1_[:,best_path[0],:],s2_[:,best_path[1],:],'replicate')
    EDsGC.append(ED)
    SDsGC.append(SD)


#%% figure 3 (c) - plotting
myFig = plt.figure(figsize=(7.5*inCm,5.75*inCm))
plt.errorbar(sigmaValues,EDsHC,yerr=SDsHC/np.sqrt(nruns*5/2),fmt='-ok',ms=4,capsize=3,label='heteroclinic channel',lw=1)
plt.errorbar(sigmaValues,EDsGC,yerr=SDsGC/np.sqrt(5*nruns/2),fmt=':sk',mfc='w',mec='k',ecolor='k',ms=4,capsize=3,label='ghost channel',lw=1)
plt.yscale('log'); plt.xscale('log')
plt.ylabel('Euclidean distance ($\mu \pm$ SEM)',fontsize=10)
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks([1e-1,1e0],fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)
plt.legend()