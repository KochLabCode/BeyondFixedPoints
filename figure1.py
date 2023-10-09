# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:01:06 2023

@author: Akhilesh Nandan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import matplotlib.pylab as pylab
import matplotlib
import matplotlib.colors as colors

import cv2
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
import os
from skimage import filters
# from numba import njit,prange,set_num_threads
import time
from skimage import measure

seed_int = np.random.randint(1000000)
# seed_int=46361
np.random.seed(seed_int)
print('random seed used ',seed_int)

#%%


def model_normalform(t,z,para):

    alpha=para[0]

    d1=alpha+z[0]**2
    d2=-z[1]
         
    return np.array([d1, d2])

def model_blackbox(t, z, para):

    d1=z[1]-(z[0]**2+0.25+alpha)
    d2=z[0]-z[1]
    
    return np.array([d1,d2])

def model_nghosts(t, z, para):
    
    alpha,k=para   
    d1=(alpha-np.sin(k*z[0]))
    d2=z[0]-z[1] 
    return np.array([d1, d2])

#%%

def estimate_eigenvalues_ana(reaction_terms,grid,eigen_threshold=None):
            
    Xg,Yg=grid
    xrange=Xg[0]
    yrange=Yg[:,0]
        
    (nrows,ncols)=np.shape(Xg)
    
    for i in range(nrows):
        for j in range(ncols):
            J=np.zeros((2,2))*np.nan
            J[0,0]=2*Xg[i,j]
            J[0,1]=0
            J[1,0]=0
            J[1,1]=-1
            
            eigen_values = np.linalg.eig(J)[0]
            
            eigen_min=np.round(eigen_values.min(),3)
            eigen_max=np.round(eigen_values.max(),3)
            
            if  eigen_threshold is None:
                Eigen_min[i,j]=eigen_min
                Eigen_max[i,j]=eigen_max
            else:
                if eigen_min<-eigen_threshold:
                    Eigen_min[i,j]=-1
                elif eigen_min>eigen_threshold:
                    Eigen_min[i,j]=1
                else:
                    Eigen_min[i,j]=0
    
                if eigen_max<-eigen_threshold:
                    Eigen_max[i,j]=-1
                elif eigen_max>eigen_threshold:
                    Eigen_max[i,j]=1
                else:
                    Eigen_max[i,j]=0
     
    return Eigen_min,Eigen_max
    


def estimate_eigenvalues(reaction_terms,grid,eigen_threshold=None):
    
    Xg,Yg=grid
    xrange=Xg[0]
    yrange=Yg[:,0]
    
    F,G=reaction_terms(t=10,z=[Xg,Yg])
    
    (nrows,ncols)=np.shape(F)
      
    Fx=np.zeros((nrows,ncols))*np.nan ## partial derivative of F w.r.t x
    for i in range(nrows): 
        Fx[i]=np.gradient(F[i],xrange)
        
    Fy=np.zeros((nrows,ncols))*np.nan ## partial derivative of F w.r.t y
    for i in range(ncols): 
        Fy[:,i]=np.gradient(F[:,i],yrange)
        
    Gx=np.zeros((nrows,ncols))*np.nan ## partial derivative of G w.r.t x
    for i in range(nrows): 
        Gx[i]=np.gradient(G[i],xrange)
    
    Gy=np.zeros((nrows,ncols))*np.nan ## partial derivative of G w.r.t y
    for i in range(ncols): 
        Gy[:,i]=np.gradient(G[:,i],yrange)
  
    Eigen_min=np.zeros((nrows,ncols))
    Eigen_max=np.zeros((nrows,ncols))
    

    for i in range(nrows):
        for j in range(ncols):
            J=np.zeros((2,2))*np.nan
            J[0,0]=Fx[i,j]
            J[0,1]=Fy[i,j]
            J[1,0]=Gx[i,j]
            J[1,1]=Gy[i,j]
            
            eigen_values = np.linalg.eig(J)[0]
            
            eigen_min=np.round(eigen_values.min(),3)
            eigen_max=np.round(eigen_values.max(),3)
            
            if  eigen_threshold is None:
                Eigen_min[i,j]=eigen_min
                Eigen_max[i,j]=eigen_max
            else:
                if eigen_min<-eigen_threshold:
                    Eigen_min[i,j]=-1
                elif eigen_min>eigen_threshold:
                    Eigen_min[i,j]=1
                else:
                    Eigen_min[i,j]=0
    
                if eigen_max<-eigen_threshold:
                    Eigen_max[i,j]=-1
                elif eigen_max>eigen_threshold:
                    Eigen_max[i,j]=1
                else:
                    Eigen_max[i,j]=0
 
    return Eigen_min,Eigen_max

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

def find_qthresh(Q,plot=None):
    '''

    Parameters
    ----------
    
    Q : Q values without any nan entries. 2D array
    
    two crucial parameters in the function are number of bins (nbins) for doing histogram
    threshold that filter out smaller bins while peak finding
    
    Returns
    -------
    qthresh : Q threshold value.
    
    found using histograms, peak finding, spline fitting and then zero crossing

    '''
    
    from scipy.signal import find_peaks
    from scipy.interpolate import CubicSpline
    
    
    
    qmin=Q.min()
    qmax=Q.max()
    
    nbins=int(1e5) # this parameter also affect the qthresh value. Need to adapt this for different models if necessary
    Qhist,bin_edges=np.histogram(Q.ravel(),bins=nbins,range=(qmin,qmax))
    
    '''
    crucial parameter that filter out peaks from the histogram.
    As a simple rule of thumb half of the maximum histogram frequency is chosen.
    So no need to adapt for each model as long as the value makes sense
    '''
    try:
        peak_thresh = 0.5*np.max(Qhist) 
        peaks, _ = find_peaks(Qhist,threshold=peak_thresh)
        qvalues=bin_edges[:-1][peaks]
        p_qvalues=Qhist[peaks]
        cs = CubicSpline(qvalues, p_qvalues)
    except:
        print('adapted threshold for peak detection')
        peak_thresh = 0.1*np.max(Qhist) 
        peaks, _ = find_peaks(Qhist,threshold=peak_thresh)
        qvalues=bin_edges[:-1][peaks]
        p_qvalues=Qhist[peaks]
        cs = CubicSpline(qvalues, p_qvalues)

    qrange=np.linspace(Q.min(),Q.max(),nbins)
    csfit=cs(qrange)
    
    csfit_grad=np.gradient(csfit,qrange[1]-qrange[0])
    zero_crossing_idx = np.min(np.where(np.diff(np.sign(csfit_grad)))[0])
    # qthresh=0.5*(qrange[0]+qrange[zero_crossing_idx])
    qthresh=np.round(qrange[zero_crossing_idx],4)
    
    if plot:
        plt.figure()
        plt.plot(bin_edges[:-1],Qhist,'-')
        plt.plot(qvalues, p_qvalues, "x")
        plt.plot(qrange,csfit,'b-')
        plt.axvline(x=qthresh,color='r')
        plt.xlabel('q values')
        plt.ylabel('#')
        plt.title('threshold = %.4f'%qthresh)
        plt.ylim(0,100)
        plt.xlim(0,qthresh+0.1)
        plt.show()

    return qthresh

def trapping_time_analytic(ll,ini,fin):
    
    import math as m
    [xin,yin]=ini
    [xfin,yfin]=fin
    if alpha<0:        
        taux=(1/ll)*(m.log(abs((2*xfin-ll)/(2*xfin+ll)))-m.log(abs((2*xin-ll)/(2*xin+ll))))
    elif alpha>0:
        taux=(2/ll)*(np.arctan(2*xfin/ll)-np.arctan(2*xin/ll))
        
    taux=np.round(taux,3)
    return taux


#%%
        
# alpha=-0.4 # saddle    
alpha=0.001 # ghost

para = [alpha]

def current_model(t,z):
    return model_normalform(t, z, para)

if alpha>0:
    xmin=-0.5;xmax=0.5
    ymin=-0.5;ymax=0.5
elif alpha<0:
    # xmin=0.2;xmax=0.7
    # ymin=-0.2;ymax=0.2
    xmin=0.13;xmax=1.13
    ymin=-0.5;ymax=0.5

xmid=np.round(np.sqrt(abs(alpha)),2)
if alpha>0:
    xmid=0
ymid=0

#%%

# alpha=0.1 # ghost

# para = [alpha]

# def current_model(t,z):
#     return model_blackbox(t, z, para)

# xmin=-0.5;xmax=1.5
# ymin=-0.5;ymax=1.5

#%%

# alpha=1.01
# k=2 
# para=[alpha,k]
# def current_model(t,z):
#     return model_nghosts(t,z,para)
# xmin=0;xmax=1.5
# ymin=0;ymax=1.5

#%%


folder_save=os.getcwd()+'svgs\\'
save_fig=None

Ng=101
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)

Xg,Yg=grid_ss

  
U,V=vector_field(current_model,grid_ss,dim='2D')    


#%%
#Kinetic evergy, Q


Q=velocity_field(current_model, grid_ss,dim='2D')   
min_idx=np.where(Q==np.min(Q)) 
Q_ov=Q.copy()

# Qthresh_aut=np.round(find_qthresh(Q_ov,plot=True),4)

Q_thresh_man=np.min(Q)+0.01
Q_thresh_man=np.round(Q_thresh_man,4)



Q_thresh=Q_thresh_man



Q_binary=Q.copy()
Q_binary[Q>=Q_thresh]=1
Q_binary[Q<Q_thresh]=0
Q[Q>=Q_thresh]=np.nan


#%%

tF=500;dt=.01;dsigma=0.0
t_eval = np.arange(0,tF,dt)
if alpha>0:
    initial_condition = [-0.5,0.2] # normal form moodel
    # initial_condition = [1,1] # black box model
    # initial_condition = [0,0.5] # black box model
elif alpha<0:
    # initial_condition = [0.47,0.2]
    initial_condition = [0.63454,0.5]
Zs=solve_timeseriesRK4(current_model,initial_condition,t_eval,dsigma,para)

x=Zs[0]
y=Zs[1]

#%%

# slow_idx=np.argwhere(Q<=Q_thresh)
slow_idx=np.argwhere(~np.isnan(Q))

Eigen_min,Eigen_max=estimate_eigenvalues(current_model,grid_ss)

Eigen_min_slow=np.zeros(np.shape(Eigen_min))*np.nan
Eigen_max_slow=np.zeros(np.shape(Eigen_max))*np.nan

for n in range(len(slow_idx)):
    xidx,yidx=[slow_idx[n][0],slow_idx[n][1]]
    Eigen_min_slow[xidx,yidx]=Eigen_min[xidx,yidx]
    Eigen_max_slow[xidx,yidx]=Eigen_max[xidx,yidx]

inCm = 1/2.54 
n=10

eig_thresh=0.7
     
fig, ax = plt.subplots(figsize=(n*8.6*inCm,n*inCm))
cm = plt.cm.get_cmap('RdYlBu')
im1=ax.imshow(Eigen_min_slow,cmap=cm,origin='lower',extent=[xmin,xmax,ymin,ymax],vmin=-eig_thresh,vmax=eig_thresh,interpolation=None)
ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)
# plt.colorbar(im1,ax=ax1,fraction=0.05)
# im2.set_alpha(0.5)
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax],fontsize=15)
ax.set_yticklabels([ymin,ymid,ymax],fontsize=15)

ax.set_title(r'$\lambda_{min}^{s}$')
ax.axhline(y=0,color='gray',ls='--')
if alpha<0:
    ax.axvline(x=xmid,color='gray',ls='--')

if save_fig:
    plt.savefig('eigen_spectrum_slow_points_alpha(%0.2f)_1.svg' %alpha,bbox_inches=0, transparent=True)
plt.show()
    
fig, ax = plt.subplots(figsize=(n*8.6*inCm,n*inCm))
cm = plt.cm.get_cmap('RdYlBu')
im2=ax.imshow(Eigen_max_slow,cmap=cm,origin='lower',extent=[xmin,xmax,ymin,ymax],vmin=-eig_thresh,vmax=eig_thresh,interpolation=None)
ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)

# plt.colorbar(im2,ax=ax,fraction=0.02)
# ax2.plot(x,y,'-',lw=4.0,color='g')

# ax2.plot(x[0],y[0],'o',ms=5.0,color='g')
# im2.set_alpha(0.5)
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax],fontsize=15)
ax.set_yticklabels([ymin,ymid,ymax],fontsize=15)

ax.set_title(r'$\lambda_{max}^{s}$')
# fig.tight_layout(pad=1.0)
# ax.set_aspect('auto')   
ax.axhline(y=0,color='gray',ls='--')
if alpha<0:
    ax.axvline(x=xmid,color='gray',ls='--')
    


if save_fig:
    plt.savefig('eigen_spectrum_slow_points_alpha(%0.2f)_2.svg' %alpha,bbox_inches=0, transparent=True)

plt.show()



#%%

qmin=1e-5
qmax=1e-1

fig, ax = plt.subplots(figsize=(n*8.6*inCm,n*inCm))
ax.streamplot(Xg,Yg,U,V,density=1,color=[0.1,0.1,0.1,0.25],arrowsize=1.5)

im2=ax.imshow(Q_ov,cmap='gnuplot2_r',origin='lower',extent=[xmin,xmax,ymin,ymax],
              norm=colors.LogNorm(vmin=qmin, vmax=qmax),interpolation=None,alpha=0.5)


# ax.scatter(Xg[min_idx],Yg[min_idx],marker='o',s=100,color='black',edgecolors='black',alpha=1)
# ax.plot(x,y,'-',lw=4.0,color='g')
# ax.plot(x[0],y[0],'o',ms=5.0,color='g')
# cbar = fig.colorbar(im2, ticks=[qmin,qmax], orientation='vertical',fraction=0.02)
# cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

# im2.set_alpha(0.25)

ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax],fontsize=15)
ax.set_yticklabels([ymin,ymid,ymax],fontsize=15)

plt.axhline(y=0,color='gray',ls='--')
if alpha<0:
    plt.axvline(x=xmid,color='gray',ls='--')
    
ax.set(xlabel='$x$', ylabel='$y$')
# # ax.set_aspect('auto')  
if save_fig: 
    plt.savefig('Qvalues_slow_points_alpha(%0.2f).svg' %alpha,bbox_inches=0, transparent=True)
plt.show()


#%%

boundary_idx=measure.find_contours(Q_binary,0)[0].astype(int)
aaidx=(boundary_idx[:,0],boundary_idx[:,1])
Q_boundary=np.zeros(np.shape(Q))
Q_boundary[aaidx]=10

fig, ax = plt.subplots(figsize=(n*8.6*inCm,n*inCm))
# ax.scatter(Xg[min_idx],Yg[min_idx],marker='o',s=100,color='black',edgecolors='black',alpha=1)
ax.plot(x,y,'-',lw=4.0,color='g')
ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)
im2=ax.imshow(Q_boundary,cmap='binary',origin='lower',extent=[xmin,xmax,ymin,ymax],vmin=0,vmax=1,interpolation=None,alpha=1)

# ax.plot(x[0],y[0],'o',ms=5.0,color='g')

# im2.set_alpha(0.25)

ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax],fontsize=15)
ax.set_yticklabels([ymin,ymid,ymax],fontsize=15)

plt.axhline(y=0,color='gray',ls='--')
if alpha<0:
    plt.axvline(x=xmid,color='gray',ls='--')
    
ax.set(xlabel='$x$', ylabel='$y$')
# # ax.set_aspect('auto')  
# plt.title(r'$Q_{thresh}=%.4f$'%Q_thresh) 

if save_fig:
    plt.savefig('Qvalues_w boundary_alpha(%0.2f).svg' %alpha,bbox_inches=0, transparent=True)

plt.show()



#%%



plot_single=None

# setting initial condition for normal form model
if alpha>0:
    minX_slow = np.min(np.round(y_range[slow_idx[:,1]],decimals=2))
    initial_condition = [1.25*minX_slow,0]
elif alpha<0:
    maxY_slow = np.max(np.round(y_range[slow_idx[:,0]],decimals=2))
    initial_condition = [np.sqrt(-alpha),1.25*maxY_slow]

# setting initial condition for blackbox model
# initial_condition = [1,1] 

# # setting initial condition for nghost model
# initial_condition = [0,0.5] 

sigmas =  [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
# sigmas =  [0.05]

nruns = 30

slowArea = []

for idx in slow_idx:
    slowArea.append(np.round([x_range[idx[1]],y_range[idx[0]]],decimals=2).tolist())

Ts = []

for dsigma in sigmas:
    print(r'$\sigma=%.4f$'%dsigma)
    T_perSigma = []
    for i in range(nruns):
        print('iteration=%i'%i)
        Zs=solve_timeseriesRK4(current_model,initial_condition,t_eval,dsigma,para)
    
        idcsTrjInSlowArea = []
        for z in range(Zs.shape[1]):
            if np.round(Zs[:,z],decimals=2).tolist() in slowArea:
                idcsTrjInSlowArea.append(z)

        idcsTrjInSlowArea = np.asarray(idcsTrjInSlowArea)
        if idcsTrjInSlowArea.size > 0:
            t0=np.min(idcsTrjInSlowArea)*dt
            t1=np.max(idcsTrjInSlowArea)*dt
            T=t1-t0
            # T = (np.max(idcsTrjInSlowArea)-np.min(idcsTrjInSlowArea))*dt
        else:
            T = 0
        T_perSigma.append(T)
        
        # if plot_single and i==2:
        if plot_single:
            x=Zs[0]
            y=Zs[1]
            
            fig, ax = plt.subplots(figsize=(5,6))
            ax.streamplot(Xg,Yg,U,V,density=1,color=[0.1,0.1,0.1,0.5])
            ax.plot(x,y,'-',ms=5.0,color='g')
            ax.imshow(Q_boundary,cmap='binary',origin='lower',extent=[xmin,xmax,ymin,ymax],vmin=0,vmax=1,interpolation=None,alpha=0.75)
            ax.imshow(Q_ov,cmap='gnuplot2_r',origin='lower',extent=[xmin,xmax,ymin,ymax],
                          norm=colors.LogNorm(vmin=1e-5, vmax=1e-1),interpolation=None,alpha=0.5)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
            if T > 0:
                ax.plot(x[np.min(idcsTrjInSlowArea):np.max(idcsTrjInSlowArea)],y[np.min(idcsTrjInSlowArea):np.max(idcsTrjInSlowArea)],'o',ms=1.0,color='r')
            # plt.title(r'$\sigma=%.4f$'%dsigma)
            plt.title(r'$T=%.4f$'%T)
            plt.show()
            
            # # inf_idxs=np.where(x>0.5)
            # inf_idxs=np.where(x<0)
            # t_eval_dummy=t_eval[:np.min(inf_idxs)]
            # x=x[:np.min(inf_idxs)]
            # y=y[:np.min(inf_idxs)]
            # Zs=Zs[:,:np.min(inf_idxs)]
            
            # fig, ax = plt.subplots(figsize=(5,6))
            # ax.plot(t_eval_dummy,x,'-',ms=5.0,color='g')
            # plt.axvline(x=t0,color='red',ls='--')
            # plt.axvline(x=t1,color='red',ls='--')
            # plt.title(r'$T=%.4f$'%T)
            # plt.show()
                        
    Ts.append(T_perSigma)
        
#%% plot 

plt.figure()
plt.subplot(1,1,1)
# for s in range(len(sigmas)):
#     plt.scatter(sigmas[s]*np.ones((nruns)),Ts[s],s=20,color='k',alpha=0.5)
plt.errorbar(sigmas,np.mean(Ts,axis=1),yerr=np.std(Ts,axis=1),color='k',capsize=5,fmt='-o',ms=8,lw=4)   
# plt.fill_between(sigmas, np.min(Ts,axis=1), np.max(Ts,axis=1),color='gray',alpha=0.1)

plt.xscale('log')
plt.xticks([1e-4,1e-3,1e-2,1e-1])
plt.xlim(sigmas[0]-0.1,sigmas[-1]+0.1)
plt.xlabel('$\sigma$',fontsize=17)
plt.ylabel('total trapping time(a.u.)')
# plt.savefig('total trapping time_(alpha=%0.2f).svg'%alpha,format='svg',bbox_inches=0, transparent=True)
plt.show()       
      
# np.save('noise intensities_(alpha=%0.2f)'%alpha,sigmas)
# np.save('trapping time_(alpha=%0.2f)'%alpha,Ts)



# myFig = plt.figure(figsize=(4*inCm,4*inCm))

# # plt.figure()
# # plt.subplot(1,1,1)
# plt.errorbar(sigmas,np.mean(Ts,axis=1),yerr=np.std(Ts,axis=1),color='k',capsize=1.5,fmt='-d',ms=3,lw=1)      
# plt.xscale('log')
# plt.xticks([1e-4,1e-3,1e-2,1e-1])
# plt.xlim(sigmas[0]-0.1,sigmas[-1]+0.1)
# plt.xlabel('$\sigma$',fontsize=17)
# plt.ylim(0,40)
# plt.ylabel('total trapping time(a.u.)')
# # plt.savefig('total trapping time_(alpha=%0.2f).svg'%alpha,format='svg',bbox_inches=0, transparent=True)
# plt.show()   

