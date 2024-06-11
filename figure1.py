# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:42:45 2023

@author: Akhilesh Nandan
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from math import isclose
import os
import matplotlib.pylab as pylab
import warnings
warnings.filterwarnings("ignore")

params = {'legend.fontsize': 15,
          'axes.labelsize': 20,
          'axes.labelpad' : 15,
          'axes.titlesize':20,
          'xtick.labelsize':20,
          'ytick.labelsize':20}
pylab.rcParams.update(params)

from quasi_potential_landscape import *

def model_ghost(t, z , para):

    alpha=para[0]

    d1=alpha+z[0]**2
    d2=-z[1]
         
    return np.array([d1, d2])

def model_saddle(t, z , para):

    alpha=para[0]

    d1=alpha+z[0]
    d2=-z[1]
         
    return np.array([d1, d2])

def model_repeller(t, z , para):

    alpha=para[0]

    d1=alpha+z[0]
    d2=z[1]
         
    return np.array([d1, d2])

def model_attractor(t, z , para):

    alpha=para[0]

    d1=alpha-z[0]
    d2=-z[1]
         
    return np.array([d1, d2])


#%%     RUN THE MODEL

"""
estimating quasi-potential from steady state probability distribution. For more details refer,

Wang, J., Xu, L., Wang, E., and Huang, S. (2010). The potential landscape of genetic circuits imposes the arrow of time 
in stem cell differentiation. Biophysical journal, 99:29–39.

"""
time_point=0
alpha=0.01

input_params=[alpha]

print('estimating quasi-potential landscape. This might take sometime.')

## change the model in the argument to plot potentials of different phase space objects.
## currently set to simulate ghost manifold using 'model_ghost' function.

qpl=QuasiPotentialLandscape(time_point,model_ghost,input_params) # for ghost landscape
# qpl=QuasiPotentialLandscape(time_point,model_saddle,input_params) # for saddle landscape

grid_pot,Pt=qpl.find_potential()  
 

U= -np.log(Pt) # quasi-potential value
zlims=[np.min(U)-0.5,np.min(U)+1]

Xpot,Ypot=grid_pot 
U=U-(np.nanmin(U)-0.5)

#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.rcParams.update({'font.size': 10})
surf = ax.plot_surface(Xpot,Ypot, U,  rstride=1, cstride=1, cmap='hot',alpha=0.3,
            linewidth=0.25,antialiased=True,edgecolor='k',vmin=np.nanmin(U),vmax=np.nanmax(U))
ax.set_aspect('auto')
ax.view_init(21,-17)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Quasi-potential')
plt.tight_layout()
plt.show()   



    

           
