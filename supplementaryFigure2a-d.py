# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

PLOTTING SUPPLEMENTARY FIGURE 2 (a-d) REQUIRES THE FILES:
- simdat_HC4_noisy_30runs.npy
- simdat_GC4_noisy_30runs.npy
TO BE IN THE SAME FOLDER AS THIS SCRIPT. 
RUN THE CODE FOR FIGURE 3(c) TO GENERATE THESE FILES. 
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

norm = plt.Normalize(0,8)
cmap=cm.get_cmap('RdPu_r')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
x0s = np.linspace(0.515,0.95,6)
y0s = (lambda y: 1 - y)(x0s)
ICs = np.asarray([x0s, y0s]).T

sigmaValues =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
nruns = 30 

#%% Supplementary figure 2 (a,b) - plotting

simulations = np.load('simdat_HC4_noisy_30runs.npy')

X = np.linspace(0,5, 20)
Y = np.linspace(0,5, 20)
grid = np.meshgrid(X, Y)

idcs = [0,3,6,9]
fig = plt.figure(figsize=(17.2*inCm,10*inCm))
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
            ax.plot(simulations[0,idx,ic,ii,1,:],simulations[0,idx,ic,ii,2,:],lw=0.5,color=col) # TODO: remove first index after rerunning
            ax.scatter(simulations[0,idx,ic,ii,1,0],simulations[0,idx,ic,ii,2,0],marker='o',color=col,s=30,edgecolors='black') # TODO: remove first index after rerunning

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
            ax.plot(simulations[0,idx,ic,ii,0,:],simulations[0,idx,ic,ii,1,:],color=col,lw=0.5) # TODO: remove first index after rerunning
    
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


#%% Supplementary figure 2 (c,d) - plotting

simulations = np.load('simdat_GC4_noisy_30runs.npy')

idcs = [0,3,6,9]

fig = plt.figure(figsize=(17.2*inCm,10*inCm))

for i in range(len(idcs)):
    plt.subplot(2,4,i+1)
    idx = idcs[i]
    ax = plt.gca()
    fun.plot_streamline(ax,models.sys_ghost4,[] ,10, grid, 1)
    ax.set_title( '$\sigma = $'+str(sigmaValues[idx]),fontsize=10)
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



