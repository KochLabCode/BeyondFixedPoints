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
import numdifftools as nd
import warnings
warnings.filterwarnings("ignore")
import cv2
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

# set_num_threads(2)
# @njit(parallel=True)
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
        infplus_idxs=np.where(Zs_prod[0]>1000)[0]
        infminus_idxs=np.where(Zs_prod[0]<-1000)[0]
        
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
alpha=0.01 # ghost

para = [alpha]

def current_model(t,z):
    return model_normalform(t, z, para)

if alpha>0:
    xmin=-0.5;xmax=0.5
    ymin=-0.5;ymax=0.5
elif alpha<0:
    xmin=0.13;xmax=1.13
    # xmin=-1.13;xmax=-0.13
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

# defining slowpoint
sp=np.array([Xg[np.where(Q_ov==np.min(Q_ov))][0],Yg[np.where(Q_ov==np.min(Q_ov))][0]])

# sometimes finding slowpoint as the minimus of Q might not be accurate bacause of
# spatial binning. It is recomended to provide with the coordinates of slowpoint manually. 


xmid=sp[0].round(3)
ymid=sp[1].round(3)

fun = lambda z: current_model(0, z)
jac = nd.Jacobian(fun)(sp)
eig_values, eig_vectors=np.linalg.eig(jac)

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
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax])
ax.set_yticklabels([ymin,ymid,ymax])

ax.set_title(r'$\lambda_{min}^{s}$')
ax.axhline(y=0,color='gray',ls='--')
if alpha<0:
    ax.axvline(x=sp[0],color='gray',ls='--')

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
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax])
ax.set_yticklabels([ymin,ymid,ymax])

ax.set_title(r'$\lambda_{max}^{s}$')
# fig.tight_layout(pad=1.0)
# ax.set_aspect('auto')   
ax.axhline(y=0,color='gray',ls='--')
# if alpha<0:
#     ax.axvline(x=xmid,color='gray',ls='--')
    


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

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax])
ax.set_yticklabels([ymin,ymid,ymax])

plt.axhline(y=0,color='gray',ls='--')
if alpha<0:
    plt.axvline(x=sp[0],color='gray',ls='--')
    
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

# fig, ax = plt.subplots(figsize=(n*8.6*inCm,n*inCm))
# # ax.scatter(Xg[min_idx],Yg[min_idx],marker='o',s=100,color='black',edgecolors='black',alpha=1)
# ax.plot(x,y,'-',lw=4.0,color='g')
# ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)
# im2=ax.imshow(Q_boundary,cmap='binary',origin='lower',extent=[xmin,xmax,ymin,ymax],vmin=0,vmax=1,interpolation=None,alpha=1)

# # ax.plot(x[0],y[0],'o',ms=5.0,color='g')

# # im2.set_alpha(0.25)

# ax.set_xlabel('x',fontsize=15)
# ax.set_ylabel('y',fontsize=15)
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(ymin,ymax)
# ax.set_xticks([xmin,xmid,xmax])
# ax.set_yticks([ymin,ymid,ymax])
# ax.set_xticklabels([xmin,xmid,xmax])
# ax.set_yticklabels([ymin,ymid,ymax])

# plt.axhline(y=0,color='gray',ls='--')
# if alpha<0:
#     plt.axvline(x=sp[0],color='gray',ls='--')
    
# ax.set(xlabel='$x$', ylabel='$y$')
# # # ax.set_aspect('auto')  
# # plt.title(r'$Q_{thresh}=%.4f$'%Q_thresh) 

# if save_fig:
#     plt.savefig('Qvalues_w boundary_alpha(%0.2f).svg' %alpha,bbox_inches=0, transparent=True)

# plt.show()



#%% Stable, unstable and ghost manifold identification

tF=50;dt=.01;dsigma=0.0 
t_eval = np.arange(0,tF,dt)


eps=0.01

Stable_manifolds=[]
Unstable_manifolds=[]
SGhost_manifolds=[]

for i in range(2): ## scan along different directions (here two directions)
    
    ## choose two initial conditions along the eigensubspace    
    initial_condition1 = sp+eps*eig_vectors[:,i]
    initial_condition2 = sp-eps*eig_vectors[:,i]
    
    ## integrating forward    
    Zs1=solve_timeseriesRK4(current_model,initial_condition1,t_eval,dsigma,para,backwards=None)
    Zs2=solve_timeseriesRK4(current_model,initial_condition2,t_eval,dsigma,para,backwards=None)
    
    

    
    ## initial distance of the trajectories to the slowpoint
    ini_dist=np.linalg.norm(eps*eig_vectors[:,i]).round(3)
    
    ## timeseries of distance when integrated forward
    dist_series1=np.sqrt(np.sum((Zs1.T-sp)**2,axis=1))
    dist_series2=np.sqrt(np.sum((Zs2.T-sp)**2,axis=1))
    fin_dist1=np.linalg.norm(Zs1[:,-1]-sp).round(3) # distance of sp to final state
    fin_dist2=np.linalg.norm(Zs2[:,-1]-sp).round(3) # distance of sp to final state
    
    mid_dist1=np.min(dist_series1).round(3) # minimum distance of the trajectory to sp
    mid_dist2=np.min(dist_series2).round(3) # minimum distance of the trajectory to sp
    
    ## for stable manifold of a hyperbolic saddle. In contrast to the  stable manifold of an
    ## attractor, the trajectories often diverge from the vicinity of the saddle.
    if (mid_dist1<ini_dist<fin_dist1 and mid_dist2<ini_dist<fin_dist2):
        
        ## integrating backwards
        Zs1_new=solve_timeseriesRK4(current_model,initial_condition1,t_eval,dsigma,para,backwards=True)
        Zs2_new=solve_timeseriesRK4(current_model,initial_condition2,t_eval,dsigma,para,backwards=True)
        
        
        # myFig = plt.figure()
        # ax =  myFig.add_subplot(1,1,1)
        # ax.streamplot(Xg,Yg,U,V,density=1,color=[0.5,0.5,0.5,0.75],arrowsize=1.5)
        # ax.plot(Zs1_new[0],Zs1_new[1],'-',color='k',lw=3.0)    
        # ax.plot(Zs2_new[0],Zs2_new[1],'-',color='k',lw=3.0)    
        # ax.set_xlim(xmin,xmax)
        # ax.set_ylim(ymin,ymax)
        # ax.set_xticks([xmin,sp[0],xmax])
        # ax.set_yticks([ymin,sp[1],ymax])
        # ax.set_xticklabels([xmin,sp[0],xmax],fontsize=15)
        # ax.set_yticklabels([ymin,sp[1],ymax],fontsize=15)   
        # plt.axhline(y=sp[1],color='gray',ls='--')
        # plt.axvline(x=sp[0],color='gray',ls='--')        
        # ax.set(xlabel='$x$', ylabel='$y$')   
        # plt.show()
            
        
        stable_mani_dummy=np.concatenate((Zs1_new,Zs2_new),axis=1)
        # stable_mani=np.sort(stable_mani_dummy,axis=1)        
        # Stable_manifolds.append(stable_mani_dummy)
        
        Stable_manifolds.append(Zs1_new)
        Stable_manifolds.append(Zs2_new)
        
    ## for stable manifold of an attractor
    elif (fin_dist1<ini_dist and fin_dist2<ini_dist): # hallmark of attraction around stable attractor
        
        ## sometimes when the grid spacing is not fine enough, the initial condition might not be
        ## exact with high precision. This might create issues with backwards integration not along 
        ## the desired direction.
        initial_condition1_new=Zs1[:,-1]+eps*eig_vectors[:,i] # this ensures the new trajs are from the steady stste
        initial_condition2_new=Zs2[:,-1]-eps*eig_vectors[:,i]
        ## integrating backwards
        Zs1_new=solve_timeseriesRK4(current_model,initial_condition1_new,t_eval,dsigma,para,backwards=True)
        Zs2_new=solve_timeseriesRK4(current_model,initial_condition2_new,t_eval,dsigma,para,backwards=True)
        
        stable_mani_dummy=np.concatenate((Zs1_new,Zs2_new),axis=1)
        # stable_mani=np.sort(stable_mani_dummy,axis=1)        
        # Stable_manifolds.append(stable_mani)
        
        Stable_manifolds.append(Zs1_new)
        Stable_manifolds.append(Zs2_new)
        
    
    ## for unstable manifold
    elif fin_dist1>mid_dist1==ini_dist and fin_dist2>mid_dist2==ini_dist:
        unstable_mani_dummy=np.concatenate((Zs1,Zs2),axis=1)
        # unstable_mani=np.sort(unstable_mani_dummy,axis=1)
        Unstable_manifolds.append(unstable_mani_dummy)
    
    ## for ghost manifold
    elif (mid_dist1<ini_dist<fin_dist1 and fin_dist2>mid_dist2==ini_dist) or (fin_dist1>mid_dist1==ini_dist and mid_dist2<ini_dist<fin_dist2):
        
        ## finding perpendicular line from which the initial conditions are initiated
        ghost_idx=np.argwhere(np.isclose(eig_values,0,atol=1e-1)==True)[0][0]
        ghost_eigv=eig_vectors[:,ghost_idx]
        # finding a vector perpendicular to the ghost eigenvector by rotating
        # the ghost eigen vector by 90 degrees.
        vector_per=ghost_eigv.copy()
        vector_per[0]=-ghost_eigv[1]
        vector_per[1]=ghost_eigv[0]
        
        dist_plane=-0.5 # determines the distance and orientation of the perpendicular line w.r.t the slow point
        [xp,yp]=sp+dist_plane*ghost_eigv
        
        # defininf the line with width delta
        delta=0.1
        xline=np.linspace(xp-delta,xp+delta,10)
        
        # inorder to find the equation for the perpendicular line, one needs to find the slope of the line.
        # this step requires division by vector_per[0]. When the ghost manifold is x axis (vector_per[0]=ghost_eigv[1]=0).
        # the condition below avoids this situation.
        if np.isclose(vector_per[0],0): 
            yline=np.linspace(yp-delta,yp+delta,10)
            xline=xp*np.ones(len(yline))
        else:
            mp=vector_per[1]/vector_per[0]
            line_per= lambda x: mp*(x-xp)+yp  
            yline=line_per(xline)
        
        # collecting all solutions from the perpendicular line
        Sols=[]
        for n in range(len(xline)):

            initial_condition = [xline[n],yline[n]]
            Zs_test=solve_timeseriesRK4(current_model,initial_condition,t_eval,dsigma,para,backwards=None)
            
            if n==0:
                lmin=len(Zs_test[0])
            
            ax.plot(Zs_test[0],Zs_test[1],'-',color='r',lw=0.5)  
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

        # funnel_idx=np.min(np.where(Dists_max<0.001)[0]) # nghost
        funnel_idx=np.min(np.where(Dists_max<0.2)[0]) # nghost
        
        Sols_cropped=[Sols[n][:,funnel_idx:] for n in range(len(Sols))]
        
        ## cropping the manifold to make it look symmteric around sp
        ghost_manifold=Sols_cropped[int(len(xline)/2)]
        dist_to_sp=np.sqrt(np.sum((ghost_manifold.T-sp)**2,axis=1))
        end_idx=np.min(np.where(dist_to_sp>dist_to_sp[0])[0])
        
        SGhost_manifolds.append(ghost_manifold[:,:end_idx])

#%% plortting identified manifolds (stable/unstable/ghost) around the slow point
 
myFig = plt.figure()
ax =  myFig.add_subplot(1,1,1)
ax.imshow(Q_boundary,cmap='binary',origin='lower',extent=[xmin,xmax,ymin,ymax],vmin=0,vmax=1,interpolation=None,alpha=1)
ax.plot(x,y,'-',lw=4.0,color='g')
# plt.arrow(fp[0],fp[1], scale_arrow*eig_vectors[0][0],scale_arrow*eig_vectors[0][1], width = width_arrow)
# plt.arrow(fp[0],fp[1], scale_arrow*eig_vectors[1][0],scale_arrow*eig_vectors[1][1], width = width_arrow)


for i in range(len(SGhost_manifolds)):
    ghost_manifold=SGhost_manifolds[i]
    ax.plot(ghost_manifold[0],ghost_manifold[1],'-',color='cyan',lw=5.0, label=r'$W^{g}$')

# if len(SGhost_manifolds)==0:
for i in range(len(Stable_manifolds)):
    stable_mani=Stable_manifolds[i]
    ax.plot(stable_mani[0],stable_mani[1],'-',color='k',lw=5.0, label=r'$W^{s}$')
    # ax.scatter(stable_mani[0],stable_mani[1],marker='.')
for i in range(len(Unstable_manifolds)):
    unstable_mani=Unstable_manifolds[i]
    ax.plot(unstable_mani[0],unstable_mani[1],'-',color='r',lw=5.0, label=r'$W^{u}$')    

# scale_arrow=5
# eig_vec1 = scale_arrow*eig_vectors[:,0]
# eig_vec2 = scale_arrow*eig_vectors[:,1]
# plt.quiver(*sp, *eig_vec1, color=['r'], scale=21)
# plt.quiver(*sp, *eig_vec2, color=['g'], scale=21)
        

ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax])
ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax])
ax.set_yticklabels([ymin,ymid,ymax])

plt.axhline(y=sp[1],color='gray',ls='--')
plt.axvline(x=sp[0],color='gray',ls='--')
    
ax.set(xlabel='$x$', ylabel='$y$')
# # ax.set_aspect('auto')  
# plt.title(r'$Q_{thresh}=%.4f$'%Q_thresh) 
plt.legend()
if save_fig:
    plt.savefig('w manifolds_alpha(%0.2f).svg' %alpha,bbox_inches=0, transparent=True)

plt.show()



#%%

"""

plot_single=None

# # setting initial condition for normal form model
# if alpha>0:
#     minX_slow = np.min(np.round(y_range[slow_idx[:,1]],decimals=2))
#     initial_condition = [1.25*minX_slow,0]
# elif alpha<0:
#     maxY_slow = np.max(np.round(y_range[slow_idx[:,0]],decimals=2))
#     initial_condition = [np.sqrt(-alpha),1.25*maxY_slow]

# setting initial condition for blackbox model
initial_condition = [1,1] 

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
alpha=0.10
# sigmas= np.load('noise intensities_(alpha=%0.2f).npy'%alpha)
# Ts=np.load('trapping time_(alpha=%0.2f).npy'%alpha)
     
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

"""
