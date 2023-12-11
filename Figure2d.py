# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:40:45 2023

@author: koch
"""



import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

from scipy.signal import find_peaks

# import cv2
params = {'legend.fontsize': 15,
          'axes.labelsize': 20,
          'axes.labelpad' : 15,
          'axes.titlesize':20,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
           'text.usetex': False,
           'font.family': 'stixgeneral',
           'mathtext.fontset': 'stix',
          
          }

pylab.rcParams.update(params)

seed_int = np.random.randint(1000000)
# seed_int=46361
np.random.seed(seed_int)
print('random seed used ',seed_int)

#%%

def model_vanderpol(t, z, para):
    mu,a,offset=para
    dx=mu*(z[0]-1/3*z[0]**3-z[1])
    dy=1/mu*z[0] + a*((z[1]+offset)-1/3*(z[1]+offset)**3)*((1+np.tanh((z[1]+offset)))/2)**10        
    return np.array([dx, dy])

def xNC(x):
    return x-x**3/3

def yNC(y,p):
    mu,a,offset = p
    return -mu*a*((y+offset)-1/3*(y+offset)**3)*((1+np.tanh(y+offset))/2)**10



#%%

def solve_timeseriesRK4(reaction_terms,initial_condition,t_eval,dsigma,stocha=None):
    
    dt=t_eval[1]-t_eval[0]
    
    ic=initial_condition
    N=len(initial_condition)
    
    Zs=np.empty((N,len(t_eval)),np.float64)
    Zs[:,0]=ic
 
    for n in range(1,len(t_eval)):
        
        zprev=Zs[:,n-1]
        
        k1=reaction_terms(t_eval[n-1],zprev)
        k2=reaction_terms(t_eval[n-1]+0.5*dt,zprev+0.5*dt*k1)
        k3=reaction_terms(t_eval[n-1]+0.5*dt,zprev+0.5*dt*k2)
        k4=reaction_terms(t_eval[n-1]+dt,zprev+dt*k3)
        
        kav=(k1+2*k2+2*k3+k4)/6

        dW=dsigma*np.sqrt(dt)*np.array([np.random.normal() for k in range(N)])
        # dW=0
        zcurr=zprev+dt*kav+dW # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
           
        Zs[:,n]=zcurr
        
    return Zs


def velocity_field(reaction_terms,grid,dim):
    
    if dim=='3D':
        Xg,Yg,Zg=grid
            
        Vx,Vy,Vz=reaction_terms(t=0,z=[Xg,Yg,Zg])
        
        Q=0.5*(Vx**2+Vy**2+Vz**2)
    elif dim=='2D':
        Xg,Yg=grid
            
        Vx,Vy=reaction_terms(t=0,z=[Xg,Yg])
        
        Q=0.5*(Vx**2+Vy**2)
    
    return Q

##################  stream line plot

# set_num_threads(2)
# @njit(parallel=True)
def vector_field(current_model,grid,dim):
  
    if dim=='3D':
        Xg,Yg,Zg=grid_ss
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        z_range=Zg[0]
        
        Lx,Ly,Lz=len(x_range),len(y_range),len(z_range)
        U=np.zeros((Lx,Ly,Lz));V=np.zeros((Lx,Ly,Lz));W=np.zeros((Lx,Ly,Lz))
        
        for i in range(Lx):
            for j in range(Ly): 
                for k in range(Lz): 
                    U[i,j,k],V[i,j,k],W[i,j,k]=current_model(0,[Xg[i,j,k],Yg[i,j,k],Zg[i,j,k]])
        
        return U,V,W
    elif dim=='2D':
        
        Xg,Yg=grid_ss
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        
        Lx,Ly=len(x_range),len(y_range)
        # U=np.zeros((Lx,Ly));V=np.zeros((Lx,Ly))
        
        U=np.empty((Lx,Ly),np.float64);V=np.empty((Lx,Ly),np.float64)
        
        for i in range(Lx):
            for j in range(Ly):  
                U[i,j],V[i,j]=current_model(0,[Xg[i,j],Yg[i,j]])
        return U,V

#%%
        
a = 0; mu = 7; offset = 0.7
para = [mu,a,offset]


def current_model(t,z):
    return model_vanderpol(t, z, para)

#full
xmin=-2.25;xmax=2.25
ymin=-1;ymax=1

xmid=0
ymid=0

Ng=151
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)

Xg,Yg=grid_ss
  
U,V=vector_field(current_model,grid_ss,dim='2D')    



#%% calculate velocity

a = 0; mu = 7; offset = 0.7
para = [mu,a,offset]

tF=10000; dt=.1;dsigma=0
t_eval = np.arange(0,tF,dt)
from matplotlib import cm
# # velocities
def euklDist(x,y): #calculates the euclidean distance between vectors x and y
    if x.shape != y.shape:
        print("Euklidean distance cannot be calculated as arrays have different dimensions")
    elif x.ndim == 1:
        EDsum = 0
        for i in range(0,x.size):
            EDsum = EDsum + np.square(x[i]-y[i])
        return np.sqrt(EDsum)
    else:
        print("Unsuitable arguments for euklidean distance calculation.")

def euklideanVelocity(x,dt):
    v = np.array([])
    n = x.shape[0]
    for i in range(1,n):
        d = euklDist(x[i,:],x[i-1,:])
        v = np.append(v, d/dt)
    return v

velocities = []
simDat = []
for a in [0,1.015]:
    initial_condition = [0.1,1]
    para = [mu,a,offset]
    Zs=solve_timeseriesRK4(current_model,initial_condition,t_eval,dsigma,para)
    velocities.append(euklideanVelocity(Zs.T, 1)/dt)
    simDat.append(Zs)

#%% Nullclines


inCm = 1/2.54; n=10



fig =  plt.subplots(1,2,figsize=(8.6*inCm,5*inCm))
plt.subplot(1,2,1)

ax = plt.gca()

#x-NC
ax.plot(x_range,xNC(x_range),'-k',lw=2)

#y-NC default
a = 0; mu = 7; offset = 0.7
para = [mu,a,offset]
ax.plot(yNC(y_range,para),y_range,'--b',lw=2)

#y-NC at SNIC
a = 1.014; mu = 7; offset = 0.7
para = [mu,a,offset]
ax.plot(yNC(y_range,para),y_range,'m',lw=2)

ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.6,0.6,0.6,1],arrowsize=0.7,linewidth=0.5)


ax.set_xlabel('x',fontsize=10)
# ax.set_ylabel('y',fontsize=10)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax],fontsize=8)
ax.set_yticklabels([ymin,ymid,ymax],fontsize=8)
    
# ax.set(xlabel='$x$')

plt.tight_layout()

# timescales and colorscale limits

# plt.subplots(figsize=(8.6*inCm,6*inCm))
h = np.asarray([])
lbls = [0,1.015]
timescales = []

hcolors = ['b','m']
plt.subplot(1,2,2)
# ax = plt.gca()

ax  = plt.gca()
for i in range(2):
    v = np.asarray(velocities[i]).flatten()
    # plt.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,alpha=0.8, histtype='step',color='k')
    histo, bin_edges = ax.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,histtype='step',color=hcolors[i],alpha=1,linewidth=0.5)[:2]
    plt.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,alpha=0.3, label = 'a='+str(lbls[i]),color=hcolors[i])
    h = np.concatenate((h, v),axis=0) 
    
    binCentres = np.array([np.mean([bin_edges[i-1],bin_edges[i]]) for i in range(1,len(bin_edges))])
    binDistance = abs(binCentres[1]-binCentres[0])


    histo = np.concatenate((np.array([0]), histo, np.array([0])))
    binCentres = np.concatenate((np.array([binCentres[0]-binDistance]), binCentres, np.array([binCentres[len(binCentres)-1]+binDistance])))
    peaks, _ = find_peaks(histo,distance=100/(i+1))
    plt.plot(binCentres[peaks],histo[peaks],'x',color=hcolors[i],ms=7,lw=15)

    tsc = 1/np.exp(binCentres[peaks])
    timescales.append(tsc)
    
    # modStr = ['Van der pol:','Van der spook:']
    # plt.text(-5,0.17-0.045*i,modStr[i])
    # for j in range(len(tsc)):
    #     plt.text(-5,0.17-0.045*i-(j+1)*0.013,'$\\tau_{'+str(j+1)+'}$='+"{:.3f}".format(tsc[j]))
      
    

# plt.legend()
plt.xlabel('log(velocity) (a.u.)',fontsize=10)
# plt.ylabel('relative frequency',fontsize=10)

plt.subplots_adjust(top=0.906,
bottom=0.343,
left=0.101,
right=0.956,
hspace=0.2,
wspace=0.399)

#Full range
plt.xticks(range(-7,3,1),fontsize=8)
plt.yticks([0,0.1,0.2],fontsize=8)


# inset 1
# plt.xlim(2.1,2.3); plt.ylim(0,.008)

# inset 2
# plt.xticks([2.1,2.3],fontsize=8)
# plt.xlim(-1.5,-1); plt.ylim(0,.025)

print('timeconstants VdP: ', timescales[0] )
print('timeconstants VdP + ghost: ', timescales[1] )



