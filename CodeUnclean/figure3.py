# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:43:08 2023

@author: koch
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import functions as fun 
import models
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))


plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

inCm = 1/2.54
stepsize = 0.05
t_end = 100     #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


#%% simulation: flow and trajectories

seeds = [2,5]

fileNames = ['data\\simdat_HC4_6ICs_s5e-3.npy','data\\simdat_HC4_6ICs_s5e-2.npy']

sigmaVals = [5e-3,5e-2]

x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

for i in range(2):   
    
    np.random.seed(seeds[i])
    
    simDat = []
    
    for ic in ICs:
        sim = fun.RK4_na_noisy(models.sys_HC4,[],ic,0,stepsize,t_end, sigmaVals[i], naFun = None,naFunParams = None)
        simDat.append(sim)
            
    simDat = np.asarray(simDat)
    simDat = np.reshape(simDat, (6,3,2000))
    
    np.save(fileNames[i], simDat)

#%% plotting: figure 3 (a)


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


#%% simulation: repeated runs from different initial conditions and different noise levels

# caution: running this simulation can take multiple hours

np.random.seed(1)

stepsize = 0.05
t_end = 500     #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)
sigmaValues =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]

nruns = 30 # only take even integers

simulations = []
#%% 
for sig in sigmaValues: 
    print(sig)
    for ic in ICs:
        for n in range(nruns):
            sim = fun.RK4_na_noisy(models.sys_HC4,[],ic,0,stepsize,t_end, sig, naFun = None,naFunParams = None)
            simulations.append(sim)
       
simulations =  np.reshape(np.asarray(simulations),(1,len(sigmaValues),len(ICs),nruns,3,timesteps))
np.save('simdat_HC4_noisy_30runs.npy',simulations)

#%% plotting: supplementary figure 2 (a,b)

simulations = np.load('simdat_HC4_noisy_30runs.npy')
sigmaValues =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
nruns = 30 

X = np.linspace(0,5, 20)
Y = np.linspace(0,5, 20)
grid = np.meshgrid(X, Y)

norm = plt.Normalize(0,8)
cmap=cm.get_cmap('RdPu_r')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T


idcs = [0,3,6,9]
fig = plt.figure(figsize=(17.2*inCm,10*inCm))
# plt.suptitle('4-saddle HC, SN val v = '+str(d),fontsize=16)
for i in range(len(idcs)):
    plt.subplot(2,4,i+1)
    idx = idcs[i]
    ax = plt.gca()
    fun.plot_streamline(ax,models.sys_HC4,[],10, grid,0.9)
    ax.set_title( '$\sigma = $'+str(sigmaValues[idx]),fontsize=10)
    ax.set_xlabel('x',fontsize=10);plt.ylabel('y',fontsize=10)
    for ic in range(5):
        col = np.asarray(cmap(norm(ic))[0:3])
        for ii in range(3):
            ax.plot(simulations[0,idx,ic,ii,1,:],simulations[0,idx,ic,ii,2,:],lw=0.5,color=col)
            ax.scatter(simulations[0,idx,ic,ii,1,0],simulations[0,idx,ic,ii,2,0],marker='o',color=col,s=30,edgecolors='black')

    ax.set_box_aspect(1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    plt.xticks([0,1,2,3,4,5])
    plt.yticks([0,1,2,3,4,5])
    
    plt.subplot(2,4,i+5)
    ax = plt.gca()
    
    plt.xlabel('time (a.u.)',fontsize=10);plt.ylabel('x',fontsize=10) 
    for ic in range(5):
        col = np.asarray(cmap(norm(ic))[0:3])
        for ii in range(3):
            ax.plot(simulations[0,idx,ic,ii,0,:],simulations[0,idx,ic,ii,1,:],color=col,lw=0.5)
    
    ax.set_box_aspect(1/2)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xlim(-10,500)
    
    plt.xticks([0,500])
    if i != 3:
        ax.set_ylim(0,6)
        plt.yticks([0,2.5,5])
    else:
        ax.set_ylim(-10,10)

plt.tight_layout()
    
plt.subplots_adjust(top=0.962,
bottom=0.038,
left=0.06,
right=0.985,
hspace=0.0,
wspace=0.41)



#%% Ghost channel

stepsize = 0.05
t_end = 300     #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


#%% flow only

seeds = [1,1]

fileNames = ['data\\simdat_GC4_6ICs_s5e-3.npy','data\\simdat_GC4_6ICs_s5e-2.npy']

sigmaVals = [5e-3,5e-2]

x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

for i in range(2):   
    
    np.random.seed(seeds[i])
    
    simDat = []
    
    for ic in ICs:
        sim = fun.RK4_na_noisy(models.sys_ghost4,[],ic,0,stepsize,t_end, sigmaVals[i], naFun = None,naFunParams = None)
        simDat.append(sim)
        ax.plot(sim[1,:],sim[2,:],lw=2.5,color=col)
        
    simDat = np.asarray(simDat)
    simDat = np.reshape(simDat, (6,3,6000))
    np.save(fileNames[i], simDat)

#%% plot flow and trajectories

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


#%% run simulations with noise

np.random.seed(1)

stepsize = 0.05
t_end = 500     #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
nruns = 30 # only take even integers

simulations_ghost = []

for sig in sigma: 
    print(sig)
    for ic in ICs:
        for i in range(nruns):
            sim = fun.RK4_na_noisy(models.sys_ghost4,[],ic,0,stepsize,t_end, sig, naFun = None,naFunParams = None)
            simulations_ghost.append(sim)
 
simulations_ghost =  np.reshape(np.asarray(simulations_ghost),(len(sigma),len(ICs),nruns,3,timesteps))
np.save('simdat_GC4_noisy_30runs.npy',simulations_ghost)

#%% plot trajectories and timecourses
simulations = np.load('simdat_GC4_noisy_30runs.npy')
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
nruns = 30 # only take even integers


#%% SFig2 GCs
norm = plt.Normalize(0,8)
cmap=cm.get_cmap('RdPu_r')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T


idcs = [0,3,6,9]

fig = plt.figure(figsize=(17.2*inCm,10*inCm))

for i in range(len(idcs)):
    plt.subplot(2,4,i+1)
    idx = idcs[i]
    ax = plt.gca()
    fun.plot_streamline(ax,models.sys_ghost4,[] ,10, grid, 1)
    ax.set_title( '$\sigma = $'+str(sigma[idx]),fontsize=10)
    ax.set_xlabel('x',fontsize=10);plt.ylabel('y',fontsize=10)
    for ic in range(5):
        col = np.asarray(cmap(norm(ic))[0:3])
        for ii in range(3):
            ax.plot(simulations[idx,ic,ii,1,:],simulations[idx,ic,ii,2,:],lw=0.5,color=col)
            ax.scatter(simulations[idx,ic,ii,1,0],simulations[idx,ic,ii,2,0],marker='o',color=col,s=30,edgecolors='black')


    ax.set_box_aspect(1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    plt.xticks([0,1,2,3,4,5])
    plt.yticks([0,1,2,3,4,5])
    
    plt.subplot(2,4,i+5)
    ax = plt.gca()
    
    plt.xlabel('time (a.u.)',fontsize=10);plt.ylabel('x',fontsize=10) 
    for ic in range(5):
        col = np.asarray(cmap(norm(ic))[0:3])
        for ii in range(3):
            ax.plot(simulations[idx,ic,ii,0,:],simulations[idx,ic,ii,1,:],color=col,lw=0.7)
    
    ax.set_box_aspect(1/2)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xlim(-10,500)
    
    plt.xticks([0,500])
    if i != 3:
        ax.set_ylim(0,6)
        plt.yticks([0,2.5,5])

plt.tight_layout()
    
plt.subplots_adjust(top=0.962,
bottom=0.038,
left=0.06,
right=0.985,
hspace=0.0,
wspace=0.41)


#%% Analyse Euclidean distances

simulationsSHC = np.load('simdat_HC4_noisy_30runs.npy')
simulationsSHC_ = np.reshape(simulationsSHC, (simulationsSHC.shape[0],simulationsSHC.shape[1],simulationsSHC.shape[2]*simulationsSHC.shape[3],simulationsSHC.shape[4],simulationsSHC.shape[5]))

simulationsGC = np.load('simdat_GC4_noisy_30runs.npy')
simulationsGC_ = np.reshape(simulationsGC, (simulationsGC.shape[0],simulationsGC.shape[1]*simulationsGC.shape[2],simulationsGC.shape[3],simulationsGC.shape[4]))

#%%
sigma =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
disp = [1]
nruns = 30 

EDsSHC = []
EDdistrSHC = []
SDsSHC = []
EDsGC = []
EDdistrGC = []
SDsGC = []

 
for iii in range(len(disp)): #
    
    EDs = []
    SDs = []
    # ds = []
    
    for i in range(len(sigma)):       
        print(i)
        
        length_ICs = 6
        
        s1 = simulationsSHC_[iii,i,:int(nruns*length_ICs/2),1:,::20]
        s1_ = np.swapaxes(s1, 1, 2)
        s2 = simulationsSHC_[iii,i,int(nruns*length_ICs/2):,1:,::20]
        s2_ = np.swapaxes(s2, 1, 2)
        
        best_path = fun.dtw_getWarpingPaths(s1_,s2_,'multiple repetitions', 'True')
    
        # ED, SD, distr = fun.euklDist_trajectory(s1_[:,best_path[0],:],s2_[:,best_path[1],:],'replicate','pairwise', meanOverReplicateDistribution = True)
        ED, SD = fun.euklDist_trajectory(s1_[:,best_path[0],:],s2_[:,best_path[1],:],'replicate')
        EDs.append(ED)
        SDs.append(SD)

    EDsSHC.append(EDs)
    SDsSHC.append(SDs)

for i in range(len(sigma)):    
    print(i)
    
    length_ICs = 6
    
    s1 = simulationsGC_[i,:int(nruns*length_ICs/2),1:,::20]
    s1_ = np.swapaxes(s1, 1, 2)
    s2 = simulationsGC_[i,int(nruns*length_ICs/2):,1:,::20]
    s2_ = np.swapaxes(s2, 1, 2)
    
    best_path = fun.dtw_getWarpingPaths(s1_,s2_,'multiple repetitions', 'True')

    ED, SD = fun.euklDist_trajectory(s1_[:,best_path[0],:],s2_[:,best_path[1],:],'replicate')
    
    EDsGC.append(ED)
    SDsGC.append(SD)

#%%
myFig = plt.figure(figsize=(7.5*inCm,5.75*inCm))
for iii in [0]: 
    plt.errorbar(sigma,EDsSHC[iii],yerr=SDsSHC[iii]/np.sqrt(nruns*5/2),fmt='-ok',ms=4,capsize=3,label='heteroclinic channel',lw=1)
plt.errorbar(sigma,EDsGC,yerr=SDsGC/np.sqrt(5*nruns/2),fmt=':sk',mfc='w',mec='k',ecolor='k',ms=4,capsize=3,label='ghost channel',lw=1)

plt.yscale('log')
plt.xscale('log')
plt.ylabel('Euclidean distance ($\mu \pm$ SEM)',fontsize=10)
plt.xticks([1e-4,1e-3,1e-2,1e-1],fontsize=8)
plt.yticks([1e-1,1e0],fontsize=8)
plt.xlabel('$\sigma$',fontsize=11)
plt.legend()
