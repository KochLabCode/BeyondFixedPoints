# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

This code reproduces the results shown in supplementary figure 4 from the study:
    
Koch D, Nandan A, Ramesan G, Tyukin I, Gorban A, Koseska A (2024): 
Ghost channels and ghost cycles guiding long transients in dynamical systems
In: Physical Review Letters (forthcoming)

IMPORTANT:
    The files "functions.py" and "models.py" need to be in the same folder as this script.
    The files:
        - data_SNIC_Farjami2021.dat
        - data_SNIC_Ghostcycle_toymodel.dat
    Need to be in the subfolder 'XPPAUT' relative to this script.
"""

# Import packages etc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import models
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions'))

import functions as fun 

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
    
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


#%% Supplementary figure 5 (a) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

with open("XPPAUT\\data_SNIC_Ghostcycle_toymodel.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat = np.asarray(data)

plt.figure(figsize=(8.6*inCm,6*inCm))

idHB=21
end_us=28
plt.plot(dat[:idHB,3],dat[:idHB,6],'-',color='peru')
plt.plot(dat[idHB-1:end_us,3],dat[idHB-1:end_us,6],':k')

id_orb1 = 29
id_orb1_end = 50
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,6],'-g')
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,9],'-g')

id_SN = 66
id_SN_end = 77
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 86
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

id_SN = 87
id_SN_end = 99
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 107
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

id_SN = 107
id_SN_end = 119
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 127
plt.plot(dat[id_SN_end-1:id_us_end,3],dat[id_SN_end-1:id_us_end,6],':k')

id_SN = 127
id_SN_end = 139
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 146
plt.plot(dat[id_SN_end-1:id_us_end,3],dat[id_SN_end-1:id_us_end,6],':k')

plt.ylim(0,2)
plt.yticks([0,0.5,1,1.5,2], fontsize=8)
plt.xlim(-0.1,1)
plt.xticks(fontsize=8)
plt.xlabel('$\\alpha$',fontsize=10)
plt.ylabel('$x$',fontsize=10)
plt.tight_layout()


#%% Supplementary figure 5 (b) 
areas = [[0,1,0,1,0,1],[1,2,0,1,0,1],[1,2,1,2,0,1],[0,1,1,2,0,1]]

steepness = 10

alphas = [1,0.2,0.002,-0.02]

print('Supplementary figure 5 (b): simulations 0% complete.')

for i in range(len(alphas)):
    alpha = alphas[i]
    
    print('Supplementary figure 5 (b): simulations ' + str(int(i*100/len(alphas))) + '% complete.')

    t_end = 100
    stepsize = 0.01
    timesteps = int(t_end/stepsize)
    timepoints = np.linspace(0, t_end, timesteps+1)
    
    ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5]] 
    
    if alpha < 0.2:
        ICs = [[0.5,0.5,0.5],[0.5,1.5,0.5],[1.5,0.5,0.5],[1.5,1.5,0.5],[1.1,1,0.5]]  
    
    myFig = plt.figure(figsize=(8,8))
    
    ax1 = myFig.add_subplot(1,1,1,projection='3d')
    ax1.set_xlim(0,2)
    ax1.set_ylim(0,2)
    ax1.set_zlim(0,2)
    
    for ic in ICs:
        sim = fun.RK4_na_noisy(models.sys_ghostCycle3D_varAlpha,[areas,steepness,alpha],ic,0,stepsize,t_end, 1e-4, naFun = None,naFunParams = None)  
        
        col = fun.euklideanVelocity(sim[1:,:].T, 1)
        cmBounds = [col.min(), col.max()]
        norm = plt.Normalize(cmBounds[0],cmBounds[1])
        cmap=cm.get_cmap('cool')
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
        simT,simX,simY,simZ = sim
        for i in range(timesteps-1):
            ax1.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=6,color=np.asarray(cmap(norm(col[i]))[0:3]))
            
    noBackground(ax1)
    ax1.view_init(60, -43)
    myFig.colorbar(sm)
    plt.tight_layout()
    
print('Supplementary figure 5 (b): plotting complete.')

#%% Supplementary figure 5 (c) 

xmin, xmax = 0, 0.8
ymin, ymax = 0.25, 0.75
zmin, zmax = 0.25, 0.75

stepsize = 0.1
t_end = 50
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


plt.figure(dpi=200)
ax = plt.axes(projection ='3d')
noBackground(ax)

alpha = -0.02

ICs = []

for x in np.linspace(xmin,xmax,7):
    for y in np.linspace(ymin,ymax,2):
        for z in np.linspace(zmin,zmax,2):
            ICs.append([x,y,z])

for ic in ICs:
    sim = fun.RK4_na_noisy(models.sys_ghostCycle3D_varAlpha,[areas,steepness,alpha],ic,0,stepsize,t_end, 1e-5, naFun = None,naFunParams = None) 
     
    simT,simX,simY,simZ = sim
    
    ax.plot3D(simX,simY,simZ ,'-',color='k',lw = 1, alpha=1)
    
    i = 0
    nexti = 0
    maxi = sim.shape[1]-1
    darrow = 0.25
    count = 0
    while i < maxi and nexti < maxi:
        d = np.linalg.norm(sim[1:,i]-sim[1:,nexti])
        if np.linalg.norm(sim[1:,i]-sim[1:,nexti]) > darrow:
            count+=1
            x0,y0,z0 = sim[1:,i]
            x1,y1,z1 = sim[1:,i+1]
            arw = Arrow3D([x0,x1],[y0,y1],[z0,z1], arrowstyle="-|>", color="k", lw = 0.5, mutation_scale=13,alpha=1)
            ax.add_artist(arw)
            i = nexti 
        nexti+=1
    
ax.set_xlabel('$x$',fontsize=18)
ax.set_ylabel('$y$',fontsize=18)
ax.set_zlabel('$z$',fontsize=18)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.view_init(43,-53)

print('Supplementary figure 5 (c): plotting complete.')

#%% Supplementary figure 5 (d) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

with open("XPPAUT\\data_SNIC_Farjami2021.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat = np.asarray(data)

plt.figure(figsize=(8.6*inCm,6*inCm))

idHB=28
end_us=65
plt.plot(dat[:idHB,3],dat[:idHB,6],'-',color='peru')
plt.plot(dat[idHB-1:end_us,3],dat[idHB-1:end_us,6],':k')

id_orb1 = 65
id_orb1_end = 224
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,6],'-g')
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,9],'-g')

id_SN = 242
id_SN_end = 312
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 333
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')


id_SN = 333
id_SN_end = 403
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 422
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

id_SN = 422
id_SN_end = 493
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-',color='peru')
id_us_end = 511
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

# plt.plot(dat[:,3],dat[:,9])

plt.ylim(1e-3,11)
plt.gca().set_yticklabels([1e-3,1e-2,1e-1,1e0,10],fontsize=8)
plt.yscale('log')
plt.xlim(0,2)
plt.xticks([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00])
plt.gca().set_xticklabels([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00],fontsize=8)
plt.xlabel('$g$',fontsize=10)
plt.ylabel('$x$',fontsize=10)
plt.tight_layout()

#%% Supplementary figure 5 (e) 
stepsize = 0.1
t_end = 460   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)

gParam = [0.2,0.5,1.51,1.6]

ICs = [
       [[0.20987, 0.1472, 2],[0.1472, 2, 0.20987], [2, 0.20987, 0.1472]],
       [[0.20987, 0.1472, 2],[0.1472, 2, 0.20987], [2, 0.20987, 0.1472]],
       [[0.20987, 0.1472, 6.5],[0.1472, 6.5, 0.20987], [6.5, 0.20987, 0.1472],[0.6,0.8,0.8]],
       [[0.20987, 0.1472, 6.5],[0.1472, 6.5, 0.20987], [6.5, 0.20987, 0.1472],[0.6,0.8,0.8]]  
       ]


print('Supplementary figure 5 (e): simulations 0% complete.')    

for n in range(len(gParam)):
    
    print('Supplementary figure 5 (e): simulations ' + str(int(n*100/len(gParam))) + '% complete.')
    g = gParam[n]
    
    myFig = plt.figure(figsize=(9,9))
    ax = myFig.add_subplot(1,1,1,projection='3d')
    
    for ic in ICs[n]:
        sim = fun.RK4_na_noisy(models.sys_Farjami2021,g,ic,0,stepsize,t_end, 1e-5, naFun = None,naFunParams = None) 
         
        simT,simX,simY,simZ = sim
        
        col = fun.euklideanVelocity(sim[1:,:].T, 1)
        cmBounds = [col.min(), col.max()]
        cmBounds = [1e-6, 0.02]
        norm = plt.Normalize(cmBounds[0],cmBounds[1])
        cmap=cm.get_cmap('cool')
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
        
        for i in range(timesteps-1):
            ax.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=6,color=np.asarray(cmap(norm(col[i]))[0:3]))
            
    ax.set_xlim(0,7)
    ax.set_ylim(0,7)
    ax.set_zlim(0,7)
    
    ax.set_xticks([0,2,4,6])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.set_yticks([0,2,4,6])
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_zticks([0,2,4,6])
    ax.zaxis.set_tick_params(labelsize=14)
     
    ax.set_xlabel('$x$',fontsize=18)
    ax.set_ylabel('$y$',fontsize=18)
    ax.set_zlabel('$z$',fontsize=18)
            
    noBackground(ax)
    ax.view_init(43,30)
    myFig.colorbar(sm)
    
print('Supplementary figure 5 (e): plotting complete.')

#%% Supplementary figure 5 (f) 

xmin, xmax = 0, 0.45
ymin, ymax = 0, 0.02
zmin, zmax = 5.5, 8

stepsize = 0.1
t_end = 50   #
timesteps = int(t_end/stepsize)
timepoints = np.linspace(0, t_end, timesteps+1)


plt.figure(dpi=200)
ax = plt.axes(projection ='3d')
noBackground(ax)

g = 1.6

ICs = []

for x in np.linspace(xmin,xmax,6):
    for y in np.linspace(ymin,ymax,2):
        for z in np.linspace(zmin,zmax,3):
            ICs.append([x,y,z])

for ic in ICs:
    sim = fun.RK4_na_noisy(models.sys_Farjami2021,g,ic,0,stepsize,t_end, 1e-5, naFun = None,naFunParams = None) 
     
    simT,simX,simY,simZ = sim
    
    ax.plot3D(simX,simY,simZ ,'-',color='k',lw = 1, alpha=1)
    
    i = 0
    nexti = 0
    maxi = sim.shape[1]-1
    darrow = 0.45
    count = 0
    while i < maxi and nexti < maxi:
        d = np.linalg.norm(sim[1:,i]-sim[1:,nexti])
        if np.linalg.norm(sim[1:,i]-sim[1:,nexti]) > darrow:
            count+=1
            x0,y0,z0 = sim[1:,i]
            x1,y1,z1 = sim[1:,i+1]
            arw = Arrow3D([x0,x1],[y0,y1],[z0,z1], arrowstyle="-|>", color="k", lw = 0.5, mutation_scale=13,alpha=1)
            ax.add_artist(arw)
            i = nexti 
        nexti+=1
    
ax.set_xlabel('$x$',fontsize=18)
ax.set_ylabel('$y$',fontsize=18)
ax.set_zlabel('$z$',fontsize=18)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.view_init(27,55)

print('Supplementary figure 5 (f): plotting complete.')
