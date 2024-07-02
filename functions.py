# -*- coding: utf-8 -*-
"""
@author: Daniel Koch

This code contains various functions for simulation and data analyses in the manuscript:
    
Koch D, Nandan A, Ramesan G, Tyukin I, Gorban A, Koseska A (2024): 
Ghost channels and ghost cycles guiding long transients in dynamical systems
In: Physical Review Letters (forthcoming)

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from dtaidistance import dtw_ndim
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis

#%% Plotting functions

def plot_bif(par,x, vline):
    
    '''
    function for plotting saddle-node bifurcation diagram
    
    '''
    
    plt.figure()
    
    if len(par)>1:
        for i in [0]: 
            plt.plot(par[i],x[i],'k-',lw=1.5) 
        for i in [1]: 
            plt.plot(par[i],x[i],'k--',lw=1.5) 
       
    else:
        plt.plot(par[0],x[0],'k-',lw=3) 
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$x^*$')
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.axhline(y=0,color='gray',ls='--',lw=1.0)
    # plt.axvline(x=0,color='gray',ls='--',lw=2.0)  
    plt.axvline(x=vline, color='b',ls='-',lw=1.0,label='\u03B1')               
    plt.legend()
    plt.show()

def plot_streamline(ax,sys,parameters,t,grid,d=2,traj=None,trajColor='m', fps=None,stab=None,save=None, **kwargs):
                
    if 'lw' in kwargs:
        lw = kwargs["lw"]
    else:
        lw = 0.7

    X=grid[0];Y=grid[1]
    
    # func = lambda u1,u2 : self.model.reaction(10,[u1,u2],self.p)
    func = lambda u1,u2 : sys([u1,u2],t,parameters)
    
    u,v=func(X,Y)
    # fig, ax = plt.subplots()
    ax.streamplot(X,Y,u,v,density=d,linewidth=lw,color=[0.66,0.66,0.66])
    if not traj is None:
        uL,uR=traj
        # ax.plot(uL[int(t/self.dt)-40:int(t/self.dt)],uR[int(t/self.dt)-40:int(t/self.dt)],lw=5.0,c='k')
        ax.plot(uL,uR,lw=0.75,c=trajColor)
        ax.scatter(uL[0],uR[0],marker='o',s=30,color=trajColor,edgecolors='black')
        
    if not fps is None:
        nn=len(fps)
        for i in range(nn):
            uLs,uRs=fps[i]
            if stab[i]=='unstable':
                color='red'
            else:
                color='black'
            ax.scatter(uLs,uRs,marker='o',s=30,color=color,edgecolors='black')

signalColor = ['grey','red','green','blue','orange','magenta']
 
#%% Other functions

def hill(x,K,nH):
    return x**nH/(x**nH + K**nH)

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

def distanceToPoint(xs, pt):
    d = np.array([])
    for i in range(xs.shape[1]):
        d = np.append(d, np.linalg.norm(xs[:,i]-pt))
    return d

def euklDist_trajectory(s1,s2, trajectoryType = 'single', mode = 'totalAvg', **kwargs):  

    ##################################################################################################################################
    # calculates the euclidean distance between trajectories s1 and s2
    # s1/s2 dimensions should be: dimensions of s1,s2: (experimental repetitions/replicates), timepoints, system/observed variables.
    # trajectoryType: 'single' or 'replicate'
    # modes for single trajectories:
    # mode for replicate trajectories: 'totalAvg', 'timeEvolution', 'totalAndtimeEvolution', 'pairwise'
    ##################################################################################################################################  
    
    if s1.shape != s2.shape:
        print('Error when calling euklDist_trajectory: array dimensions do not match!')
        return
    
    if trajectoryType == 'replicate' and (mode == 'totalAvg' or 'timeEvolution' or 'totalAndtimeEvolution'):
        
        reps = s1.shape[0]
        nTimePts = s1.shape[1]

        EDs = np.zeros((reps,nTimePts))
    
        # calculate euclidean distances over replicates and time
        for i in range(reps):
            for ii in range(nTimePts):
                    EDs[i,ii] = euklDist(s1[i,ii,:], s2[i,ii,:])

        ED_mean_or = np.mean(EDs,axis=0) # mean of EDs across repetitions at specified timepts
        ED_SD_or = np.std(EDs,axis=0)  # SD of EDs across repetitions at specified timepts
        
        # Endpoint values for full trajectories
        
        ED_mean_otr = np.mean(EDs) # mean over time and repetitions
        ED_SD_otr = (np.mean(ED_SD_or**2))**0.5 # SD over time and repetitions
        
        # Evolution of mean ED and SD across repetitions up until time t for all timepoint t
        
        ED_tevol = np.array([])
        SD_tevol = np.array([])
        
        for t in range(1,nTimePts):
            ED_tevol = np.append(ED_tevol, np.mean(ED_mean_or[:t]))
            SD_tevol = np.append(SD_tevol, (np.mean(ED_SD_or[:t]**2))**0.5)
            
        if mode == 'totalAvg':
            return ED_mean_otr, ED_SD_otr
        elif mode == 'timeEvolution':
            return ED_tevol, SD_tevol
        elif mode == 'totalAndtimeEvolution':
            return ED_mean_otr, ED_SD_otr, ED_tevol, SD_tevol
        
    if trajectoryType == 'replicate' and mode == 'pairwise':
        
        reps = s1.shape[0]
             
        ED_mean_ot = []
        ED_SD_ot = []
        
        # calculate euclidean distances over replicates and time
        for i in range(reps):
            bp = dtw_getWarpingPaths(s1[i,:,:],s2[i,:,:],'single repetition')
            EDs_ = []
            for ii in range(bp[0].shape[0]):
                    EDs_.append(euklDist(s1[i,bp[0][ii],:], s2[i,bp[1][ii],:]))
                
            ED_mean_ot.append(np.mean(EDs_))
            ED_SD_ot.append(np.std((EDs_)))
  
        # Endpoint values for full trajectories
        
        ED_mean_otr = np.mean(ED_mean_ot) # mean over time and repetitions
        ED_SD_otr = (np.mean(np.asarray(ED_SD_ot)**2))**0.5 # SD over time and repetitions
        
        if 'meanOverReplicateDistribution' in kwargs:
            if kwargs['meanOverReplicateDistribution'] == True:
                return ED_mean_otr, ED_SD_otr, ED_mean_or
            else:
                return ED_mean_otr, ED_SD_otr
        else:
            return ED_mean_otr, ED_SD_otr
        
    if trajectoryType == 'single' and (mode == 'totalAvg' or 'timeEvolution'):
        
        nTimePts = s1.shape[0]
        
        EDs = np.zeros(nTimePts)
    
        # calculate euclidean distances over time
        for i in range(nTimePts):
            EDs[i,] = euklDist(s1[i,:], s2[i,:])
        
        
        # Endpoint values for full trajectories
        ED_mean = np.mean(EDs,axis=0)
        ED_SD = np.std(EDs,axis=0)
        
        # Evolution of mean ED and SD across repetitions up until time t for all timepoint t
        
        ED_tevol = np.array([])
        SD_tevol = np.array([])
        
        for t in range(1,nTimePts):
            
            ED_tevol = np.append(ED_tevol, np.mean(EDs[:t]))
            SD_tevol = np.append(SD_tevol, np.std(EDs[:t]))
        if mode == 'totalAvg':
            return ED_mean, ED_SD
        elif mode == 'timeEvolution':
            return ED_tevol, SD_tevol
        
        

def dtw_getWarpingPaths(s1,s2, mode = 'multiple repetitions', showWarpingPaths = False, **kwargs): 
    # print('DTW in')
    ##################################################################################################################################
    # This function performs a dynamic time warping alignment of n-dimensional trajectories s1 and s2
    # dimensions of s1,s2: (experimental repetitions), timepoints, system/observed variables.
    # The time-warping itself is done by dtaidistance package, see https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html 
    ##################################################################################################################################
    
    if kwargs.get('mode') != None:
        mode == kwargs['mode']
    
    if kwargs.get('showWarpingPaths') != None:
        showWarpingPaths == kwargs['showWarpingPaths']
        
    if mode == 'multiple repetitions':
        avg_s1 = np.mean(s1,axis=0); avg_s2 = np.mean(s2,axis=0)
    elif mode == 'single repetition': 
        avg_s1 = s1; avg_s2 = s2;
    else:
        print('Unknown argument for mode')
        
    d, paths = dtw_ndim.warping_paths(avg_s1, avg_s2)
    best_path = dtw.best_path(paths)
    
    if showWarpingPaths == True:
        fig = plt.figure(figsize=(6.5, 6))
        dtwvis.plot_warpingpaths(avg_s1, avg_s2, paths, best_path, figure=fig)
        plt.show()
    # print('DTW out')
    return np.asarray(best_path)[:,0], np.asarray(best_path)[:,1]


def allinfo_filereader_2var(folder_load,filename,threshold):
    
    '''
    
    this function loads saved .dat file from xppaut and splits the steady state values
    into different branches (stable and unstable). Currently customized to 
    two dimensional systems.
    
    inputs
    --------
    
    folder_load: fpath to folder where the datfile is saved
    filename: name of the .dat file
    threshold: number of points that specify whether a branch needs to 
               be considered or not.
   
    returns
    --------
    
    p: nested list of parameter values. Each list within correspond to different branches.
    ss_u1: nested list of steady state values of first variable
    ss_u2: nested list of steady state values of second variable

    '''
    
    df = pd.read_table(os.path.join(folder_load,filename+'.dat'), sep="\s+",header=None,skiprows=1)
    df=df.to_numpy()    
    limit=-1
    
    par=df[:,3][:limit]
    u1=df[:,6][:limit]
    u2=df[:,7][:limit]
    
    
    branch=df[:,0][:limit]
    inds=branch[:-1]-branch[1:]
    branch_cut_inds=np.argwhere(inds!=0)+[[1]]
    
    if len(branch_cut_inds)==0:
        ss_u1=[u1]
        ss_u2=[u2]
        p=[par]
        return p,ss_u1,ss_u2
    else:
        ss_u1=[u1[:branch_cut_inds[0][0]]];ss_u2=[u2[:branch_cut_inds[0][0]]] # first branch
        p=[par[:branch_cut_inds[0][0]]]
        
        for i in range(len(branch_cut_inds)-1): 
            if len(u1[branch_cut_inds[i][0]:branch_cut_inds[i+1][0]])<=threshold:
                pass
            else:
                ss_u1.append(u1[branch_cut_inds[i][0]:branch_cut_inds[i+1][0]])
                ss_u2.append(u2[branch_cut_inds[i][0]:branch_cut_inds[i+1][0]])
                p.append(par[branch_cut_inds[i][0]:branch_cut_inds[i+1][0]])
        
        try:
            ss_u1.append(u1[branch_cut_inds[i+1][0]:-1]) # last branch
            ss_u2.append(u2[branch_cut_inds[i+1][0]:-1])
            p.append(par[branch_cut_inds[i+1][0]:-1])
        except:
            ss_u1.append(u1[branch_cut_inds[0][0]:-1]) # last branch
            ss_u2.append(u2[branch_cut_inds[0][0]:-1])
            p.append(par[branch_cut_inds[0][0]:-1])

        
        return p,ss_u1,ss_u2

def load_allinfo_file(folder_load, filename):
    par,x_ss,y_ss = allinfo_filereader_2var(folder_load,filename,threshold=5)
    return par,x_ss,y_ss


#%% Integrators
  
def RK4_na_noisy(f,p,ICs,t0,dt,t_end, sigma=0, naFun = None,naFunParams = None):     # args: ODE system, parameters, initial conditions, starting time t0, dt, number of steps
        steps = int((t_end-t0)/dt)
        x = np.zeros([steps,len(ICs)])
        t = np.zeros(steps,dtype=float)
        x[0,:] = ICs
        t[0] = t0
        
        if naFun != None and naFunParams != None:
            for i in range(1,steps):
                
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1,:],t[i-1],p,naFun,naFunParams)*dt
                k2 = f(x[i-1,:]+k1/2,t[i-1],p,naFun,naFunParams)*dt
                k3 = f(x[i-1,:]+k2/2,t[i-1],p,naFun,naFunParams)*dt
                k4 = f(x[i-1,:]+k3,t[i-1],p,naFun,naFunParams)*dt
                x_next = x[i-1,:] + (k1+2*k2+2*k3+k4)/6
                dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape[0]) # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x[i,:] = x_next + dW
        else:
            for i in range(1,steps):
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1,:],t[i-1],p)*dt
                k2 = f(x[i-1,:]+k1/2,t[i-1],p)*dt
                k3 = f(x[i-1,:]+k2/2,t[i-1],p)*dt
                k4 = f(x[i-1,:]+k3,t[i-1],p)*dt
                x_next = x[i-1,:] + (k1+2*k2+2*k3+k4)/6
                dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape[0]) # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x[i,:] = x_next + dW
            
        return np.vstack((t,x.T))
    

def RK4_na_noisy_pos(f,p,ICs,t0,dt,t_end, sigma=0, naFun = None,naFunParams = None):     # args: ODE system, parameters, initial conditions, starting time t0, dt, number of steps
        steps = int((t_end-t0)/dt)
        x = np.zeros([steps,len(ICs)])
        t = np.zeros(steps,dtype=float)
        x[0,:] = ICs
        t[0] = t0
        
        if naFun != None and naFunParams != None:
            for i in range(1,steps):
                
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1,:],t[i-1],p,naFun,naFunParams)*dt
                k2 = f(x[i-1,:]+k1/2,t[i-1],p,naFun,naFunParams)*dt
                k3 = f(x[i-1,:]+k2/2,t[i-1],p,naFun,naFunParams)*dt
                k4 = f(x[i-1,:]+k3,t[i-1],p,naFun,naFunParams)*dt
                x_next = x[i-1,:] + (k1+2*k2+2*k3+k4)/6
                dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape[0]) # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x_ = x_next + dW
                x_[x_<0] = 0
                x[i,:] = x_
        else:
            for i in range(1,steps):
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1,:],t[i-1],p)*dt
                k2 = f(x[i-1,:]+k1/2,t[i-1],p)*dt
                k3 = f(x[i-1,:]+k2/2,t[i-1],p)*dt
                k4 = f(x[i-1,:]+k3,t[i-1],p)*dt
                x_next = x[i-1,:] + (k1+2*k2+2*k3+k4)/6
                dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape[0]) # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x_ = x_next + dW
                x_[x_<0] = 0
                x[i,:] = x_

        return np.vstack((t,x.T))

#%% functions for the fokker-planck solver

import enum
from inspect import getfullargspec

def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(value)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def slice_idx(i, ndim, s0):
    """return a boolean array for a ndim-1 slice along the i'th axis at value s0"""
    idx = [slice(None)]*ndim
    idx[i] = s0

    return tuple(idx)

def combine(*funcs):
    """combine a collection of functions into a single function (for probability, potential, and force functions)"""
    def combined_func(*args):
        values = funcs[0](*args)
        for func in funcs[1:]:
            values += func(*args)

        return values

    return combined_func

class boundary(enum.Enum):
    """enum for the types ofboundary conditions"""
    reflecting = enum.auto()
    periodic   = enum.auto()

def vectorize_force(f):
    """decorator to vectorize a force function"""
    ndim = len(getfullargspec(f).args)
    signature = ','.join(['()']*ndim)
    signature += '->(N)'

    vec_f = np.vectorize(f, signature=signature)
    def new_func(*args):
        return np.rollaxis(vec_f(*args), axis=-1, start=0)

    return new_func

