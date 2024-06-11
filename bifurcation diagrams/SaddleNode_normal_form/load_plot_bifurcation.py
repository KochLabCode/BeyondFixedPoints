# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:00:40 2021

@author: nandan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pylab as pylab

params = {'legend.fontsize': 15,
          'axes.labelsize': 20,
          'axes.labelpad' : 15,
          'axes.titlesize':20,
          'xtick.labelsize':20,
          'ytick.labelsize':20}
pylab.rcParams.update(params)

#%%

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

def load_allinfo_file():
    filename='SN_normalform_allinfo'
    par,x_ss,y_ss = allinfo_filereader_2var(os.getcwd(),filename,threshold=5)
    return par,x_ss,y_ss

def plot_bif(par,x):
    
    plt.figure()
    
    if len(par)>1:
        for i in [0]: 
            plt.plot(par[i],x[i],'k-',lw=3) 
        for i in [1]: 
            plt.plot(par[i],x[i],'k--',lw=3) 
       
    else:
        plt.plot(par[0],x[0],'k-',lw=3) 
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$x^*$')
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.axhline(y=0,color='gray',ls='--',lw=2.0)
    plt.axvline(x=0,color='gray',ls='--',lw=2.0)
    if save_fig==True:
        plt.savefig(os.path.join(folder_save,'SN_normal form.svg'))                    
    plt.show()

#%%
    
save_fig=None   
folder_save=os.path.abspath(os.getcwd())
par,x_ss,y_ss= load_allinfo_file()
plot_bif(par,x_ss)


