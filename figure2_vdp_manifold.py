# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:20:03 2023

@author: nandan
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numdifftools as nd
from scipy.spatial import distance
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore")

params = {'legend.fontsize': 15,
          'axes.labelsize': 20,
          'axes.labelpad' : 15,
          'axes.titlesize':20,
          'xtick.minor.size': 3,
          'xtick.major.width': 2.15,
          'xtick.minor.width': 1.25,
          'ytick.major.size': 10,
          'ytick.minor.size': 3,
          'ytick.major.width': 2.15,
          'ytick.minor.width': 1.25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'xtick.major.size':10,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
            'axes.linewidth': 2,
          
          }

pylab.rcParams.update(params)


def model_normalform(t,z,para):

    alpha=para[0]

    d1=alpha+z[0]**2
    d2=-z[1]
         
    return np.array([d1,d2])

def model_blackbox(t, z, para):

    d1=z[1]-(z[0]**2+0.25+alpha)
    d2=z[0]-z[1]
    
    return np.array([d1,d2])

def model_nghosts(t, z, para):
    alpha,k=para   
    d1=(alpha-np.sin(k*z[0]))
    d2=z[0]-z[1] 
    return np.array([d1, d2])


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




def solve_timeseriesRK4(reaction_terms,initial_condition,t_eval,dsigma,stocha=None,backwards=None):
    
    dt=t_eval[1]-t_eval[0]
    
    if backwards==True:
        dt=-dt
    
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

        dW=dsigma*np.sqrt(abs(dt))*np.array([np.random.normal() for k in range(N)])
        # dW=0
        
        zcurr=zprev+dt*kav+dW # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
           
        Zs[:,n]=zcurr
    
    try:
        Zs_prod=np.array([Zs[i]*Zs[i+1] for i in range(len(initial_condition)-1)])
        nan_idxs=np.where(np.isnan(Zs_prod[0]))[0]
        infplus_idxs=np.where(Zs_prod[0]==np.inf)[0]
        infminus_idxs=np.where(Zs_prod[0]==-np.inf)[0]
        
        if len(nan_idxs)>0:
            nan_min_indx=np.min(nan_idxs)
        else:
            nan_min_indx=np.inf
            
        if len(infplus_idxs)>0:
            infplus_min_indx=np.min(infplus_idxs)
        else:
            infplus_min_indx=np.inf
            
        if len(infminus_idxs)>0:
            infminus_min_indx=np.min(infminus_idxs)
        else:
            infminus_min_indx=np.inf
            
        min_idx=np.min([nan_min_indx,infplus_min_indx,infminus_min_indx])
        Zs=Zs[:,:int(min_idx)]
    except:
        pass  
    return Zs


def vector_field(current_model,grid):
         
    Xg,Yg=grid
    
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
    
# # alpha=-0.4 # saddle    
# alpha=0.01 # ghost
# para = [alpha]
# def current_model(t,z):
#     return model_normalform(t, z, para)
# xmin=-1;xmax=1
# ymin=-1;ymax=1
# scale_arrow=0.5;width_arrow=0.02
# fp=np.array([0,0])

# delta=0.001;A=0.2
# xghost=np.linspace(fp[0]-0.1,fp[0]+0.1,100)
# yghost_ana=delta*A*np.exp(-np.arctan(xghost/np.sqrt(alpha))/np.sqrt(alpha))

# ynullcline_normalform= lambda x: np.zeros(len(x_range))

   

# alpha=0.01 # ghost
# para = [alpha]
# def current_model(t,z):
#     return model_blackbox(t, z, para)
# xmin=-0.5;xmax=1.5
# ymin=-0.5;ymax=1.5
# scale_arrow=0.5;width_arrow=0.01
# fp=np.array([0.5,0.5])

# delta=0.01;A=0.1
# xghost=np.linspace(fp[0]-0.5,fp[0]+0.5,50)
# yghost_ana=xghost+delta*A*np.exp(np.arctan((2*xghost-1)/(2*np.sqrt(alpha)))/np.sqrt(alpha))

# ynullcline_blackbox= lambda x: x_range    


# alpha=1.01
# k=1 
# para=[alpha,k]
# def current_model(t,z):
#     return model_nghosts(t,z,para)
# xmin=0;xmax=3
# ymin=0;ymax=3
# initial_condition = [-1,0]
# scale_arrow=2;width_arrow=0.05
# fp=np.array([np.pi/2,np.pi/2])

# delta=0.001;A=1
# xghost=np.linspace(fp[0]-0.1,fp[0]+0.1,10)
# yghost_ana=xghost+delta*A*np.exp((2/np.sqrt(alpha**2-1))*np.arctan((alpha*np.tan(xghost/2)-1)/(np.sqrt(alpha**2-1))))


a = 0; mu = 7; offset = 0.7
para = [mu,a,offset]

def current_model(t,z):
    return model_vanderpol(t, z, para)

xmin=-2.25;xmax=2.25
ymin=-1;ymax=1
initial_condition = [-0.5,0.5]
scale_arrow=2;width_arrow=0.05
sp=np.array([-1.971,0.597])


#%%

Ng=101
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)

Xg,Yg=grid_ss

U,V=vector_field(current_model,grid_ss)    

#%%

tF=100;dt=.01;dsigma=0.0 
t_eval = np.arange(0,tF,dt)

# fp=np.array([-np.sqrt(-alpha),0])


eps=-0.2

fun = lambda z: model_vanderpol(0, z, para=[7, 1.014,0.7])
jac = nd.Jacobian(fun)(sp)
eig_values, eig_vectors=np.linalg.eig(jac)

## finding perpendicular line
# ghost_idx=np.argwhere(np.isclose(eig_values,0,atol=1e-1)==True)[0][0]
ghost_idx=np.argmax(eig_values)
ghost_eigv=eig_vectors[:,ghost_idx]
vector_per=ghost_eigv.copy()
vector_per[0]=-ghost_eigv[1]
vector_per[1]=ghost_eigv[0]

[xp,yp]=sp+eps*ghost_eigv
delta=0.1
xline=np.linspace(xp-delta,xp+delta,10)

if np.isclose(vector_per[0],0):
    yline=np.linspace(yp-delta,yp+delta,10)
    xline=xp*np.ones(len(yline))
else:
    mp=vector_per[1]/vector_per[0]
    line_per= lambda x: mp*(x-xp)+yp  
    yline=line_per(xline)

#%%       

# ynullcline=ynullcline_normalform(x_range)
# ynullcline=ynullcline_blackbox(x_range)
xnullcline=xNC(x_range)
ynullcline_default=yNC(y_range,para)


a = 1.014
para = [mu,a,offset]
ynullcline_snic=yNC(y_range,para)

#%%
myFig = plt.figure()
ax =  myFig.add_subplot(1,1,1)
# ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)

# plt.arrow(fp[0],fp[1], eps*eig_vectors[0][0],eps*eig_vectors[0][1], width = width_arrow, color='r')
# plt.arrow(fp[0],fp[1], eps*eig_vectors[1][0],eps*eig_vectors[1][1], width = width_arrow, color='b')

ax.plot(xline,yline,'-',lw='5',color='k')

Sols=[]

for n in range(len(xline)):

    initial_condition = [xline[n],yline[n]]
    Zs_test=solve_timeseriesRK4(current_model,initial_condition,t_eval,dsigma,para,backwards=None)
    
    if n==0:
        lmin=len(Zs_test[0])
    
    ax.plot(Zs_test[0],Zs_test[1],'-',color='g',lw=0.5)  
    ax.plot(Zs_test[0][0],Zs_test[1][0],'.',color='g')  
    Sols.append(Zs_test)
    if len(Zs_test[0])<lmin:
        lmin=len(Zs_test[0])

Dists=[]
ref_traj=Sols[0][:,:lmin]
# Dists=np.zeros(())
for n in range(1,len(Sols)):
    Zs=Sols[n][:,:lmin]
    dist=np.sqrt(np.sum((Zs-ref_traj)**2,axis=0))
    Dists.append(dist)
    
Dists=np.reshape(Dists,(len(Sols)-1,lmin))
Dists_max=np.max(Dists,axis=0)

funnel_idx=np.min(np.where(Dists_max<np.min(Dists_max)+0.02)[0]) # nghost

ax.plot(x_range,xnullcline,'-k',lw=2)
ax.plot(ynullcline_default,y_range,'--b',lw=2)
ax.plot(ynullcline_snic,y_range,'m',lw=2)

# ax.plot(ynullcline,y_range,'k',lw=3.0,label=r'$y_0(x)$')
# ax.plot(xghost,yghost_ana,'-',color='cyan',lw=3.0,label=r'$W_{loc}^g(x)$')  

plt.axvline(x=sp[0],ls='--',color='gray')
plt.axhline(y=sp[1],ls='--',color='gray')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

# ax.set_xlim(-0.1,0.1)
# ax.set_ylim(-0.1,0.1)

# ax.set_xlim(-0.0005,0.0005)
# ax.set_ylim(-0.001,0.001)

plt.legend()
ax.set_xlabel('x',fontsize=20)
ax.set_ylabel('y',fontsize=20)
plt.tight_layout()
plt.show()

#%% selecting ghost manifold

Sols_cropped=[Sols[n][:,funnel_idx:] for n in range(len(Sols))]


## cropping the manifold to make it look symmteric around sp
ghost_manifold=Sols_cropped[int(len(xline)/2)]
dist_to_sp=np.sqrt(np.sum((ghost_manifold.T-sp)**2,axis=1))
end_idx=np.min(np.where(dist_to_sp>dist_to_sp[0])[0])

ghost_manifold=ghost_manifold[:,:end_idx]

# myFig = plt.figure()
# ax =  myFig.add_subplot(1,1,1)
# ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)
# ax.plot(ghost_manifold[0],ghost_manifold[1],'-',color='cyan',lw=3.0)  
# plt.axvline(x=sp[0],ls='--',color='gray')
# plt.axhline(y=sp[1],ls='--',color='gray')
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(ymin,ymax)
# ax.set_xlabel('x',fontsize=20)
# ax.set_ylabel('y',fontsize=20)
# plt.tight_layout()
# plt.show()


#%%
# inCm = 1/2.54 
# n=10

# fig, ax = plt.subplots(figsize=(n*8.6*inCm,n*inCm))
myFig = plt.figure()
ax =  myFig.add_subplot(1,1,1)
ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)

# plt.arrow(fp[0],fp[1], eps*eig_vectors[0][0],eps*eig_vectors[0][1], width = width_arrow, color='r')
# plt.arrow(fp[0],fp[1], eps*eig_vectors[1][0],eps*eig_vectors[1][1], width = width_arrow, color='b')

# ax.plot(xline,yline,'-',lw='5',color='k')

ax.plot(x_range,xnullcline,'-k',lw=2)
ax.plot(ynullcline_default,y_range,'--b',lw=2)
ax.plot(ynullcline_snic,y_range,'m',lw=2)

ax.plot(ghost_manifold[0],ghost_manifold[1],'-',color='cyan',lw=3.0)  

# ax.plot(ynullcline,y_range,'k',lw=3.0,label=r'$y_0(x)$')
# ax.plot(xghost,yghost_ana,'-',color='cyan',lw=3.0,label=r'$W_{loc}^g(x)$')  

# plt.axvline(x=sp[0],ls='--',color='gray')
# plt.axhline(y=sp[1],ls='--',color='gray')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

# ax.set_xlim(-0.1,0.1)
# ax.set_ylim(-0.1,0.1)

# ax.set_xlim(-0.0005,0.0005)
# ax.set_ylim(-0.001,0.001)

plt.legend()
ax.set_xlabel('x',fontsize=20)
ax.set_ylabel('y',fontsize=20)
plt.tight_layout()
plt.show()

#%%


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

#%%

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

ax.plot(ghost_manifold[0],ghost_manifold[1],'-',color='cyan',lw=3.0)  

ax.set_xlabel('x',fontsize=10)
# ax.set_ylabel('y',fontsize=10)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmax])
ax.set_yticks([ymin,ymax])
ax.set_xticklabels([xmin,xmax],fontsize=8)
ax.set_yticklabels([ymin,ymax],fontsize=8)
    
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
plt.xticks(range(-7,3,1),fontsize=8)
plt.yticks([0,0.1,0.2],fontsize=8)

plt.subplots_adjust(top=0.906,
bottom=0.343,
left=0.101,
right=0.956,
hspace=0.2,
wspace=0.399)


# plt.xlim(2.1,2.3); plt.ylim(0,.008)
# plt.xticks([2.1,2.3],fontsize=8)
plt.xlim(-1.5,-1); plt.ylim(0,.025)
# plt.savefig('modified van der Pol_figure2.svg',bbox_inches=0, transparent=True)
plt.show()


q = 5
q1 = np.percentile(h, q)
q2 = np.percentile(h, 100-q)
q3 = np.percentile(h, 98)
# plt.vlines([q1,q2], 0,1, color = 'r')
# plt.tight_layout()

cmBounds = [q1, q2]

print('limits for colorbar: ', cmBounds)
 
norm = plt.Normalize(cmBounds[0],cmBounds[1])





