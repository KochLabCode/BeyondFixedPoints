# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

This code reproduces the results shown in supplementary figure 6 from the study:
    
Koch D, Nandan A, Ramesan G, Tyukin I, Gorban A, Koseska A (2024): 
Ghost channels and ghost cycles guiding long transients in dynamical systems
In: Physical Review Letters (forthcoming)

IMPORTANT:
    The files "functions.py" and "models.py" need to be in the same folder as this script.
    The files:
        - data_SNIC_hybrid_lowerHB.dat
        - data_SNIC_hybrid_upperHB.dat
        - data_SNIC_hybrid_SN.dat
    Need to be in the subfolder 'XPPAUT' relative to this script.
"""

# Import packages etc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions'))

import functions as fun 
import models

# settings for plotting
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    
#%% Supplementary figure 6 (a)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

area = [[0,1,0,1,0,1],[1,2,0,1,0,1],
         [0,1,1,2,0,1],[1,2,1,2,0,1],[2,3,1,2,0,1],
         [1,2,2,3,0,1],[2,3,2,3,0,1]]
s = 10
alpha = 0.002

t_end = 400
stepsize = 0.01
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


# set random seed (optional)
seed_int = 6
np.random.seed(seed_int)

print('Supplementary figure 6 (a): simulate...')
simDat = fun.RK4_na_noisy(models.sys_hybrid,[area,s,alpha],[2,2,0.5],0,stepsize,t_end, 1e-4, naFun = None,naFunParams = None)  
velocities = fun.euklideanVelocity(simDat[1:,:].T, 1)/stepsize
print('Supplementary figure 6 (a): simulation complete.')

# calculate color bounds
q = 5
q1 = np.percentile(velocities, q)
q2 = np.percentile(velocities, 100-q)
q3 = np.percentile(velocities, 98)

cmBounds = [q1, q2]

norm = plt.Normalize(cmBounds[0],cmBounds[1])
cmap=cm.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


# plot
myFig = plt.figure(figsize=(4.5*inCm,7*inCm))
ax = myFig.add_subplot(1,1,1,projection='3d')

simT,simX,simY,simZ = simDat
d = velocities*stepsize
t = 0
t_last = 0
while t < timesteps-1:
    if sum(d[t_last:t]) > q3*stepsize:
        ax.plot3D([simX[t_last],simX[t]],[simY[t_last],simY[t]],[simZ[t_last],simZ[t]] ,'-',lw=2, color=np.asarray(cmap(norm(velocities[t]))[0:3]))
        t_last = t
    t+=1

axLims = [0.4,2.7]
axTicks = [0.5,1.5,2.5]

noBackground(ax)
ax.view_init(39,69)
ax.set_xlim(axLims)
ax.set_ylim(axLims)
ax.set_zlim(axLims)

ax.set_xticks(axTicks)
ax.xaxis.set_tick_params(labelsize=8)
ax.set_yticks(axTicks)
ax.yaxis.set_tick_params(labelsize=8)
ax.set_zticks(axTicks)
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
print('Supplementary figure 6 (a): plotting complete.')

#%% Supplementary figure 6 (b)

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
stateTCs[0,:] = simDat[0,::nth]

for i in range(len(points)):
    dist = fun.distanceToPoint(simDat[1:,::nth],points[i])
    stateTCs[i+1,:] = dist

myFig = plt.figure(figsize=(7*inCm,6*inCm))
ax1 = myFig.add_subplot(1,1,1)
for i in range(len(points)):
    if i != 3:
        myColor = cm.get_cmap('seismic',7)(i)
        lst = 'solid'
    else:
        myColor = 'black'
        lst = 'dashed'
    ax1.plot(stateTCs[0,:], fun.hill(stateTCs[i+1,:],0.3,-3) ,linestyle=lst, label=str(i+1), color = myColor, lw=1.2) 

ax1.set_xlabel('time (a.u.)',fontsize=10)
ax1.set_ylabel('$g_1,g_2,g_3,g_4,g_5,g_6$',fontsize=10)
ax1.set_box_aspect(1/3)
ax1.set_yticks([0,.5,1])
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

print('Supplementary figure 6 (b): plotting complete.')

#%% Supplementary figure 6 (c)

with open("XPPAUT\\data_SNIC_hybrid_lowerHB.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

datHB_low = np.asarray(data)

with open("XPPAUT\\data_SNIC_hybrid_upperHB.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

datHB_up = np.asarray(data)

with open("XPPAUT\\data_SNIC_hybrid_SN.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

datSN = np.asarray(data)


#plotting

plt.figure(figsize=(4.5*inCm,4*inCm))

#plot lower HB
idHB=20
end_us=28
plt.plot(datHB_low[:idHB,3],datHB_low[:idHB,6],'-',color='peru')
plt.plot(datHB_low[idHB-1:end_us,3],datHB_low[idHB-1:end_us,6],'--k')

id_orb1 = 28
id_orb1_end = 64
plt.plot(datHB_low[id_orb1:id_orb1_end,3],datHB_low[id_orb1:id_orb1_end,6],'-g')
plt.plot(datHB_low[id_orb1:id_orb1_end,3],datHB_low[id_orb1:id_orb1_end,9],'-g')

#plot upper HB
idHB=20
end_us=28
plt.plot(datHB_up[:idHB,3],datHB_up[:idHB,6],'-',color='peru')
plt.plot(datHB_up[idHB-1:end_us,3],datHB_up[idHB-1:end_us,6],'--k')

id_orb1 = 28
id_orb1_end = 64
plt.plot(datHB_up[id_orb1:id_orb1_end,3],datHB_up[id_orb1:id_orb1_end,6],'-g')
plt.plot(datHB_up[id_orb1:id_orb1_end,3],datHB_up[id_orb1:id_orb1_end,9],'-g')

#plot SNs

id_SN = 1
id_SN_end = 12
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 22
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 22
id_SN_end = 34
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 42
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 42
id_SN_end = 54
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 62
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 62
id_SN_end = 74
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 82
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 82
id_SN_end = 94
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 102
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 102
id_SN_end = 114
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 229
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')


plt.ylim(0,3)
plt.yticks([0,1,2,3], fontsize=8)
plt.xlim(-0.1,0.6)
plt.xticks([0,0.25,0.5], fontsize=8)
plt.xticks(fontsize=8)
plt.xlabel('$\\alpha$',fontsize=10)
plt.ylabel('$x$',fontsize=10)
plt.tight_layout()

print('Supplementary figure 6 (c): plotting complete.')