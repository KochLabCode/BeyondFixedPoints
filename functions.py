# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:33:11 2022

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random as rnd
import networkx as nx
import glob
from PIL import Image
import os
import datetime
from scipy import interpolate
from dtaidistance import dtw_ndim
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

#%% Plotting functions

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

def plot_tc_nw_sigs(x,ys,signals,t_end,stepsize,**kwargs):
    
    if 'labels' in kwargs: 
        lbls = kwargs['labels']
    else: lbls = []
    
    if 'scalingfactor' in kwargs: 
        sF = kwargs['scalingfactor']
    else: sF = 1

    for i in range(min(len(signals),5)):
        print(signals)
        signal_profile = sig2array(signals[i],t_end,stepsize)
        # print(signal_profile.shape)
        plt.fill_between(np.linspace(0,t_end,signal_profile.shape[0])*sF,signal_profile,color=signalColor[i],alpha=0.2,label='s'+str(i))
        
    if np.ndim(ys) > 1:
        n = min(ys.shape)
        for i in range(n):
            if len(lbls)>0:
                plt.plot(x*sF,ys[i],label=lbls[i])
            else:
                plt.plot(x*sF,ys[i])
    else:   
        plt.plot(x,ys,label=lbls)
            
    plt.legend()
    plt.xlabel(kwargs['xlabel'],fontsize=13)
    plt.ylabel(kwargs['ylabel'],fontsize=13)
    
def plot_tc_phasespace_sig(ax,dat,idcs,signals,t_end,stepsize,**kwargs):
    
    # fig = argfig
    # decode kwargs and default values
        
    # saveFig = kwargs.get('saveFig',False)
    # saveAnimation = kwargs.get('saveAnimation',False)    
    # frameDur = kwargs.get('frameDur',5)
    # axlabels = kwargs.get('axlabels',['x','y','z'])
    # path = kwargs.get('path',os.getcwd()+'\\'+fileName)
    # linecolor = kwargs.get('color','k')
    
    if 'saveFig' in kwargs:
        saveFig = kwargs['saveFig']
    else: 
        saveFig = False    
    
    if 'saveAnimation' in kwargs:
        saveAnimation = kwargs['saveAnimation']
    else: 
        saveAnimation = False 
        
    if 'frameDur' in kwargs:
        frameDur = kwargs['frameDur']
    else: 
        frameDur = 5
        
    if 'axlabels' in kwargs:
        axlabels = kwargs['axlabels']
    else:
        axlabels = ['x','y','z']   
        
        
    if 'color' in kwargs:
        linecolor = kwargs['color']
    else: 
       linecolor = 'k'
        
    if saveFig == True or saveAnimation == True:
        
        if 'fileName' in kwargs:
           fileName = kwargs['fileName']
        else:
            fileName = str(datetime.datetime.now())
            fileName = fileName.replace(' ','_')
            fileName = fileName.replace(':','-')
            fileName = fileName.replace('.','-')
        
        if 'path' in kwargs:
            path = kwargs['path']
        else: 
            path = os.getcwd()+'\\'+fileName
            
        if not os.path.exists(path):
            os.makedirs(path)
            
    #determine transparency by sum of all signals
    
    signal_profiles = np.zeros((len(signals),max(dat.shape)))   
    for i in range(len(signals)):
        signal_profiles[i,:] = sig2array(signals[i],t_end,stepsize)
    signal_tot = np.sum(signal_profiles,axis=0)
    signal_tot = signal_tot/max(signal_tot)
    
    alpha_ch = 0.1 + 0.9*signal_tot
    
    # plot phase space trajectory
    # ax = fig.gca(projection="3d")
    # ax = fig.add_subplot(projection='3d')
    
    idx_end = max(dat.shape)
    for i in range(idx_end-1):
        plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=linecolor,alpha=alpha_ch[i],linewidth=3)

    plt.plot(dat[idcs[0],0], dat[idcs[1],0], dat[idcs[2],0],'ow',mec='k', ms=8)
    plt.plot(dat[idcs[0],idx_end-1], dat[idcs[1],idx_end-1], dat[idcs[2],idx_end-1],'xk', ms=8)
    ax.grid(False)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    ax.set_zlabel(axlabels[2])
    
    if saveFig == True:
        plt.savefig(os.path.join(path,fileName+'.png'),dpi=300, bbox_inches = "tight")     
        
    # video animation  
    if saveAnimation == True:   
    
        xlims=ax.get_xlim()
        ylims=ax.get_ylim()
        zlims=ax.get_zlim()
        fs=plt.gcf().get_size_inches()  
        
        for j in range(idx_end):
            newFig = plt.figure(figsize=fs)
            ax = newFig.add_subplot(projection='3d')
            # ax = newFig.gca(projection="3d")   
            for i in range(j-1):
                plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=linecolor,alpha=alpha_ch[i],linewidth=3)           
            plt.plot(dat[idcs[0],0], dat[idcs[1],0], dat[idcs[2],0],'ok',mec='k', ms=8)
            plt.plot(dat[idcs[0],j], dat[idcs[1],j], dat[idcs[2],j],'ow',mec='k', ms=8)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.grid(False)
            plt.xlabel(axlabels[0])
            plt.ylabel(axlabels[1])
            ax.set_zlabel(axlabels[2])
            plt.savefig(os.path.join(path,str(j)+'.png'),dpi=100, bbox_inches = "tight")
            plt.close(newFig)
                    
        # filepaths
        fp_in = os.path.join(path,"*.png")
        fp_out = os.path.join(path,fileName+".gif")
        
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) ) ]
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=frameDur, compress_level=9)
        

def plot_tc_phasespace_colored(ax,dat,idcs,**kwargs): 
    
    # fig = argfig
    # decode kwargs and default values
        
    # saveFig = kwargs.get('saveFig',False)
    # saveAnimation = kwargs.get('saveAnimation',False)    
    # frameDur = kwargs.get('frameDur',5)
    # axlabels = kwargs.get('axlabels',['x','y','z'])
    # path = kwargs.get('path',os.getcwd()+'\\'+fileName)
    # linecolor = kwargs.get('color','k')
    
    if 'saveFig' in kwargs:
        saveFig = kwargs['saveFig']
    else: 
        saveFig = False    
    
    if 'saveAnimation' in kwargs:
        saveAnimation = kwargs['saveAnimation']
    else: 
        saveAnimation = False 
        
    if 'frameDur' in kwargs:
        frameDur = kwargs['frameDur']
    else: 
        frameDur = 5
        
    if 'axlabels' in kwargs:
        axlabels = kwargs['axlabels']
    else:
        axlabels = ['x','y','z']   
        
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else: 
       mode = 'time'
           
    if 'colormap' in kwargs:
        colm = kwargs['colormap']
    else: 
       colm = 'plasma'
       
    if 'input' in kwargs:
        inp = kwargs['input']
        plotInput = True
        if 'alpha_p' in kwargs:
            alpha_p = kwargs['alpha_p']
        else: 
            alpha_p = 0.05
    else: 
        plotInput = False
        
        
    if 'fileName' in kwargs:
       fileName = kwargs['fileName']
    else:
        fileName = str(datetime.datetime.now())
        fileName = fileName.replace(' ','_')
        fileName = fileName.replace(':','-')
        fileName = fileName.replace('.','-')
            
    if saveFig == True or saveAnimation == True:
                
        if 'path' in kwargs:
            path = kwargs['path']
        else: 
            path = os.getcwd()+'\\'+fileName
            
        if not os.path.exists(path):
            os.makedirs(path)
            
    #determine transparency by sum of all signals
    
    # signal_profiles = np.zeros((len(signals),max(dat.shape)))   
    # for i in range(len(signals)):
    #     signal_profiles[i,:] = sig2array(signals[i],t_end,stepsize)
    # signal_tot = np.sum(signal_profiles,axis=0)
    # signal_tot = signal_tot/max(signal_tot)
    
    # alpha_ch = 0.1 + 0.9*signal_tot
    
    # plot phase space trajectory
    # ax = fig.gca(projection="3d")
    # ax = fig.add_subplot(projection='3d')
    
    if mode == 'velocity':
        col = euklideanVelocity(dat.T, 1)
    elif mode == 'time':
        col = np.asarray(range(max(dat.shape)))
    
    print(fileName,' min/max velocity:',col.min(),col.max())
    
    if 'cmBounds' in kwargs:
        cmBounds = kwargs['cmBounds']
    else: 
        cmBounds = [col.min(), col.max()]
    
    norm = plt.Normalize(cmBounds[0],cmBounds[1])
    cmap=cm.get_cmap(colm)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    idx_end = max(dat.shape)
    # print(idx_end,col.shape)

    
    
    if plotInput == True:
        for i in range(0,idx_end-1,1):
                if inp[i] >0:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2]+0.05,'-',color='m',alpha=1,linewidth=2)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                else:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=alpha_p,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                    
    else:
        for i in range(idx_end-1):
            plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
        
        plt.plot(dat[idcs[0],0], dat[idcs[1],0], dat[idcs[2],0],'ow',mec='k', ms=8)
        plt.plot(dat[idcs[0],idx_end-1], dat[idcs[1],idx_end-1], dat[idcs[2],idx_end-1],'xk', ms=8)
    
    ax.grid(False)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    ax.set_zlabel(axlabels[2])
    # plt.gcf().colorbar(sm,ticks=[col.min(),(col.min()+col.max())/2,col.max()])
    
    if saveFig == True:
        plt.savefig(os.path.join(path,fileName+'.png'),dpi=300, bbox_inches = "tight")     
        
    # video animation  
    if saveAnimation == True:   
    
        xlims=ax.get_xlim()
        ylims=ax.get_ylim()
        zlims=ax.get_zlim()
        fs=plt.gcf().get_size_inches()  
        
        for j in range(idx_end):
            newFig = plt.figure(figsize=fs)
            ax = newFig.add_subplot(projection='3d')
            # ax = newFig.gca(projection="3d")   
            for i in range(j-1):
                # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                if inp[i] >0:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2]+0.05,'-',color='m',alpha=1,linewidth=2)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                else:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=alpha_p,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
            # plt.plot(dat[idcs[0],0], dat[idcs[1],0], dat[idcs[2],0],'ok',mec='k', ms=8)
            # plt.plot(dat[idcs[0],j], dat[idcs[1],j], dat[idcs[2],j],'ow',mec='k', ms=8)
            # ax.view_init(elev=20, azim=150)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.grid(False)
            plt.xlabel(axlabels[0])
            plt.ylabel(axlabels[1])
            ax.set_zlabel(axlabels[2])
            plt.savefig(os.path.join(path,str(j)+'.png'),dpi=100, bbox_inches = "tight")
            plt.close(newFig)
                    
        # filepaths
        fp_in = os.path.join(path,"*.png")
        fp_out = os.path.join(path,fileName+".gif")
        
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) ) ]
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=frameDur, compress_level=9)

def plot_tc_phasespace(ax,dat,idcs,**kwargs): 
    
    # fig = argfig
    # decode kwargs and default values
        
    # saveFig = kwargs.get('saveFig',False)
    # saveAnimation = kwargs.get('saveAnimation',False)    
    # frameDur = kwargs.get('frameDur',5)
    # axlabels = kwargs.get('axlabels',['x','y','z'])
    # path = kwargs.get('path',os.getcwd()+'\\'+fileName)
    # linecolor = kwargs.get('color','k')
    
    if 'saveFig' in kwargs:
        saveFig = kwargs['saveFig']
    else: 
        saveFig = False    
    
    if 'saveAnimation' in kwargs:
        saveAnimation = kwargs['saveAnimation']
    else: 
        saveAnimation = False 
        
    if 'frameDur' in kwargs:
        frameDur = kwargs['frameDur']
    else: 
        frameDur = 5
        
    if 'axlabels' in kwargs:
        axlabels = kwargs['axlabels']
    else:
        axlabels = ['x','y','z']   
        
           
    if 'color' in kwargs:
        col = kwargs['color']
    else: 
      col = 'C0'
       
    if 'input' in kwargs:
        inp = kwargs['input']
        plotInput = True
        if 'alpha_p' in kwargs:
            alpha_p = kwargs['alpha_p']
        else: 
            alpha_p = 0.05
    else: 
        plotInput = False
        
        
    if 'fileName' in kwargs:
       fileName = kwargs['fileName']
    else:
        fileName = str(datetime.datetime.now())
        fileName = fileName.replace(' ','_')
        fileName = fileName.replace(':','-')
        fileName = fileName.replace('.','-')
            
    if saveFig == True or saveAnimation == True:
                
        if 'path' in kwargs:
            path = kwargs['path']
        else: 
            path = os.getcwd()+'\\'+fileName
            
        if not os.path.exists(path):
            os.makedirs(path)
            

    idx_end = max(dat.shape)
    # print(idx_end,col.shape)


    if plotInput == True:
        for i in range(0,idx_end-1,1):
                if inp[i] >0:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2]+0.05,'-',color='m',alpha=1,linewidth=2)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=col,alpha=1,linewidth=3)
                else:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=alpha_p,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=col,alpha=alpha_p,linewidth=3)
                    
    else:
        for i in range(idx_end-1):
            plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=col,alpha=1,linewidth=3)
        
        plt.plot(dat[idcs[0],0], dat[idcs[1],0], dat[idcs[2],0],'ow',mec='k', ms=8)
        plt.plot(dat[idcs[0],idx_end-1], dat[idcs[1],idx_end-1], dat[idcs[2],idx_end-1],'xk', ms=8)
    
    ax.grid(False)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    ax.set_zlabel(axlabels[2])
    # plt.gcf().colorbar(sm,ticks=[col.min(),(col.min()+col.max())/2,col.max()])
    
    if saveFig == True:
        plt.savefig(os.path.join(path,fileName+'.png'),dpi=300, bbox_inches = "tight")     
        
    # video animation  
    if saveAnimation == True:   
    
        xlims=ax.get_xlim()
        ylims=ax.get_ylim()
        zlims=ax.get_zlim()
        fs=plt.gcf().get_size_inches()  
        
        for j in range(idx_end):
            newFig = plt.figure(figsize=fs)
            ax = newFig.add_subplot(projection='3d')
            # ax = newFig.gca(projection="3d")   
            for i in range(j-1):
                # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                if inp[i] >0:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=1,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2]+0.05,'-',color='m',alpha=1,linewidth=2)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=col,alpha=1,linewidth=3)
                else:
                    # plt.plot(dat[idcs[0],[i,i+1]], dat[idcs[1],[i,i+1]], dat[idcs[2],[i,i+1]],'-',color=np.asarray(cmap(norm(col[i]))[0:3]),alpha=alpha_p,linewidth=3)
                    plt.plot(dat[idcs[0],i:i+2], dat[idcs[1],i:i+2], dat[idcs[2],i:i+2],'-',color=col,alpha=alpha_p,linewidth=3)
            # plt.plot(dat[idcs[0],0], dat[idcs[1],0], dat[idcs[2],0],'ok',mec='k', ms=8)
            # plt.plot(dat[idcs[0],j], dat[idcs[1],j], dat[idcs[2],j],'ow',mec='k', ms=8)
            # ax.view_init(elev=20, azim=150)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.grid(False)
            plt.xlabel(axlabels[0])
            plt.ylabel(axlabels[1])
            ax.set_zlabel(axlabels[2])
            plt.savefig(os.path.join(path,str(j)+'.png'),dpi=100, bbox_inches = "tight")
            plt.close(newFig)
                    
        # filepaths
        fp_in = os.path.join(path,"*.png")
        fp_out = os.path.join(path,fileName+".gif")
        
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) ) ]
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=frameDur, compress_level=9)
        
def plotSignals(ax, sig, mode='steps', t_end = None, resolution = 400, xlims=None, ylims=None, **kwargs):
    if t_end == None:
        t_end = sig[0][sig[0].shape[0]-1]+1
    
    if mode == 'steps':
        x = np.arange(0,t_end,t_end/resolution)
        y = signal2array(sig,t_end,t_end/resolution)
        ax.step(x,y,**kwargs)
        if ylims != None:
            ax.set_ylim(ylims[0], ylims[1])
        if xlims != None:
            ax.set_xlim(xlims[0], xlims[1])
            
    if mode == 'events':
        cmap = cm.get_cmap('RdYlBu')
        normSig = (sig[1] - np.min(sig[1])) / (np.max(sig[1]) - np.min(sig[1]))
        if not 'color' in kwargs:
            for i in range(0,len(sig[0]),2):
                rectangle = plt.Rectangle((sig[0][i],0.1), sig[0][i+1]-sig[0][i], 0.8, fc=cmap(normSig[int(i/2)]),ec = 'k',**kwargs)
                ax.add_patch(rectangle)
        else:
            for i in range(0,len(sig[0]),2):
                rectangle = plt.Rectangle((sig[0][i],0.1), sig[0][i+1]-sig[0][i], 0.8,**kwargs)
                ax.add_patch(rectangle)
            
        if ylims != None:
            ax.set_ylim(ylims[0], ylims[1])
        if xlims != None:
            ax.set_xlim(xlims[0], xlims[1])
        ax.set_yticks([])
            


#%% Euclidean distance functions and dynamic time warping

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

def orthogonality(x,y):
    if x.shape != y.shape:
        print("Orthogonality cannot be calculated as vectors have different dimensions")
    else:
        o = np.array([])
        n = x.shape[0]
        for i in range(n):
            o = np.append(o, 1/np.sum(x[i,:]*y[i,:]))
        return o
    
def euklDist_TvP(trj, p):
    
    nTimePts = trj.shape[0]

    EDs = np.zeros(nTimePts)

    # calculate euclidean distances over replicates and time
    for i in range(nTimePts):
        EDs[i] = np.linalg.norm(trj[i,:] - p)
        # euklDist(trj[i,:], p)
        
    return EDs


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

def interpAlign(v,method='linear'): #input v should be tuple. For each item, first dimension should be time. First column should be timepoints followed by x1,...,xn 
    
    # Requirements:
    # from scipy import interpolate
    # import numpy as np
    
    t_steps = np.asarray([v[0].shape[0], v[1].shape[0]])
    n_intp = np.max(t_steps)
    idx_v_intp = np.where(n_intp != t_steps)[0][0]
    xdims = v[idx_v_intp].shape[1]-1

    t_start = v[idx_v_intp][:,0][0]
    t_stop = v[idx_v_intp][:,0][v[idx_v_intp].shape[0]-1]
    
    #interpolate time
    xnew = np.linspace(t_start,t_stop,n_intp) 
    
    #interpolate x1...xn
    ynew = []
    for i in range(0,xdims):
        f = interpolate.interp1d(v[idx_v_intp][:,0], v[idx_v_intp][:,i+1],kind=method)
        ynew.append(f(xnew))
    
    ynew = np.asarray(ynew)
    ynew = np.reshape(ynew,(xdims,n_intp))
    
    vnew = np.vstack((xnew,ynew)).T
    
    if idx_v_intp == 0:
        vRet = [vnew,v[1]]
    else:
        vRet = [v[0],vnew]
    return vRet
    
def repeatedInterpAlign(v,method='linear'):
    
    # after dynamic time warping the number of timepoints between two trajectories can be different when comparing to a third trajectory from another class over the same time.
    # To allow cross-class comparison, this function interpolates trajectories to achieve the same number of datapoints for all trajectories.

    reps = v[0].shape[0]
    x1 = v[0].shape[1]
    x2 = v[1].shape[1]
    xpts = np.max((v[0].shape[1],v[1].shape[1]))
    ndims = v[0].shape[2]
    
    x1vec = np.linspace(1,x1,x1)
    x2vec = np.linspace(1,x2,x2)
    
    y1 = np.array([])
    y2 = np.array([])
    for i in range(reps):
        
        s1 = np.empty((x1,ndims+1))
        s1[:,0] = x1vec
        s1[:,1:] = v[0][i,:,:]
        
        s2 = np.empty((x2,ndims+1))
        s2[:,0] = x2vec
        s2[:,1:] = v[1][i,:,:]
        
        if x1 != x2:
            s1, s2 = interpAlign((s1,s2))
        
        y1 = np.append(y1,s1[:,1:])
        y2 = np.append(y2,s2[:,1:])
        
    y1 = np.reshape(y1,(reps,xpts,ndims))
    y2 = np.reshape(y2,(reps,xpts,ndims))
    return y1,y2

#%% Other functions

def igc(m,lower,upper,nd): #information gain complexity v1.0
    # m: 2D pattern, lowest values for states, upper values for states, 
    # nd: number of states from lowest to highest
    
    if m.ndim != 2: 
        print('Error: Matrix m has unsuitable number of dimensions!')
        return
    else:
        nr = m.shape[0] # nr or rows
        nc = m.shape[1] # nr or columns
    
    bounds = np.linspace(lower,upper,nd+1)[1:] # state boundaries
    
    def state(x,bnds): #returns the state to which a value x belongs
        return np.min(np.where(x<=bnds)) 
    
    # transform the 2D pattern m into a matrix with nd states
    ms = np.zeros(m.shape) 
    
    for k in range(nr):
        for l in range(nc):
            ms[k,l] = state(m[k,l],bounds)
    
    ms = np.asarray(ms,dtype=int)
    
    def neighborSubset(s):  #returns the (upper) neighbor of all elements in the state matrix ms which are in state s
        subidcs = np.where(ms==s)
        subset = []
        for i in range(len(subidcs[0])): 
            if subidcs[0][i] >= 1:
                idr = subidcs[0][i]
                idc = subidcs[1][i]
                subset.append(m[idr-1,idc])
        return np.asarray(subset,dtype=int)
    
    # individual state probabilities
    p = []
    for i in range(nd):
        p.append(np.count_nonzero(ms == i))
    p = np.asarray(p)/ms.size
    
    #joint and conditional probabilities
    p_joint = np.zeros((nd,nd))
    p_cond = np.zeros((nd,nd))
    
    for i in range(nd):
        sub_i = neighborSubset(i)
        for j in range(nd):
            p_cond[i,j] = np.count_nonzero(sub_i == j)
            if sub_i.size != 0: 
                p_cond[i,j] = p_cond[i,j]/sub_i.size
            p_joint[i,j]=p[i]*p_cond[i,j]
                
    #information gain complexity
    G = 0
    
    for i in range(nd):
        for j in range(nd):
            if p_cond[i,j] != 0:
                G += p_joint[i,j]*np.log2(p_cond[i,j])
        
    return -G, p,p_joint,p_cond

def fromIntv(t,intv):
    for i in range(0,len(intv),2):
        if t>=intv[i] and t<=intv[i+1]:
            return 1
    return 0

def readSignals(t,s):
    intv, a = s
    for i in range(0,len(intv),2):
        if t>=intv[i] and t<=intv[i+1]:
            return a[int(i/2)]
    return 0

def sigGenFun(nr_of_pulses, pulse_dur, pause_dur, p_shuffle, t_first, t_end): 
    ###!!!!! OUTDATED - use 'generateSignals' instead !!!!!###
    s = []
    for i in range(1,nr_of_pulses+1):
        dice = rnd.random()
        if dice < p_shuffle:
            s_start = t_first+(i-1)*(pulse_dur+pause_dur)
            s_stop = t_first+(i-1)*(pulse_dur+pause_dur)+pulse_dur
            s.append(s_start)
            s.append(s_stop)
        else:
            new_pos = rnd.randint(0, t_end)
            s_start = new_pos 
            s_stop = new_pos+pulse_dur
            s.append(s_start)
            s.append(s_stop)           
    s.sort()
    return np.array(s)


def generateSignals(nr_of_pulses, pulse_dur, pause_dur, p_shuffle, t_first, t_end, p1 = 1, mode = 'binary',**kwargs):
    
    if 'seed' in kwargs:
        np.random.seed(kwargs['seed'])
    
    s = []
    for i in range(1,nr_of_pulses+1):
        dice = rnd.random()
        if dice < p_shuffle:
            s_start = t_first+(i-1)*(pulse_dur+pause_dur)
            s_stop = t_first+(i-1)*(pulse_dur+pause_dur)+pulse_dur
            s.append(s_start)
            s.append(s_stop)
        else:
            new_pos = rnd.randint(0, t_end)
            s_start = new_pos 
            s_stop = new_pos+pulse_dur
            s.append(s_start)
            s.append(s_stop)           
    s.sort()
    
    if p1 != 1 or mode == 'binary':
        a = []
        for i in range(nr_of_pulses):
            dice = rnd.random()
            if dice < p1:
                a.append(1)
            else:
                a.append(-1)
        np.random.seed()
        return [np.array(s), np.array(a)]
    elif mode == 'continuous':
        a = []
        for i in range(nr_of_pulses):
            dice = rnd.random()
            if dice < p1:
                a.append(rnd.random())
            else:
                a.append(-rnd.random())
        np.random.seed()
        return [np.array(s), np.array(a)]
    elif mode == 'alternating':
        a = []
        for i in range(nr_of_pulses):
            dice = rnd.random()
            if i%2==0:
                a.append(1)
            else:
                a.append(-1)
        return [np.array(s), np.array(a)]

def simDat_frag_receptorsOnly(simDat,rdim,sdim,timesteps):
    sdR = simDat[:rdim,:]  
    sdP = simDat[rdim:2*rdim,:]
    sdLRa = np.reshape(simDat[2*rdim:,:],(rdim,sdim,timesteps))
    return sdR, sdP, sdLRa

def simDat_frag(simDat,rdim,sdim,timesteps):
    sdR = simDat[:rdim,:]  
    sdP = simDat[rdim:2*rdim,:]
    sdLRa = np.reshape(simDat[2*rdim:2*rdim+rdim*sdim],(rdim,sdim,timesteps))
    X = simDat[2*rdim+rdim*sdim:]
    return sdR, sdP, sdLRa, X

def sig2array(sig,t,sz): 
    length = int(t/sz)
    a = np.array([])
    for i in range(length):
        s = 0
        for ii in range(0,len(sig),2):
            if i*sz >= sig[ii] and i*sz <= sig[ii+1]:
                s = 1
        a = np.append(a,s)
    return 

def signal2array(sig,t_end,sz):
    intvs, amps = sig
    length = int(t_end/sz)
    a = np.array([])
    for i in range(length):
        s = 0
        for ii in range(0,len(intvs),2):
            if i*sz >= intvs[ii] and i*sz <= intvs[ii+1]:
                s = amps[int(ii/2)]
        a = np.append(a,s)
    return a

def paramBL(baseVal, adjM):
    dim = adjM.shape[0]
    v = np.ones([dim,1])*baseVal
    for i in range(0,dim):
        if adjM[i,:].sum() > 0:
            v[i] = v[i]/adjM[i,:].sum()
    return v

def genRndNet(dim, pa = 0.85, pi = 0.15):

    p_act = pa/dim
    p_inh = pi/dim

    adj_matrix = np.zeros([dim,dim])
      
    for i in range(0,dim):
        for ii in range(0,dim):
            dice = rnd.random()
            if dice <= p_act:
                adj_matrix[i][ii] = 1
            dice = rnd.random()
            if dice <= p_inh:
                adj_matrix[i][ii] = -1

    if dim>1:
        for i in range(0,dim):
            while np.all(adj_matrix[i,:]==0) or (adj_matrix[i,i]==1 and sum(adj_matrix[i,:]) == 1):
                dice = rnd.random()
                pos = i
                while pos == i:
                    pos = rnd.randint(0,dim-1)
                if dice < p_act:
                    adj_matrix[i][pos] = 1
                dice = rnd.random()
                if dice <= p_inh:
                   adj_matrix[i][pos] = -1 
                   
    return adj_matrix

def drawNetwork(adj_matrix,vRout):
    nw_dim = adj_matrix.shape[0]
    G = nx.from_numpy_matrix(adj_matrix.transpose(),parallel_edges = True, create_using = nx.DiGraph)
    pos = nx.spring_layout(G, k=10/np.sqrt(nw_dim),iterations = 100)
    posFullNet=pos
    for j in range(len(vRout)):
        posFullNet["Input "+str(j)]=np.array([0+j,-1])
        G.add_node("Input "+str(j))

    inhEdges = []
    actEdges = []
    for j in range(len(vRout)):
        for i in np.where(np.asarray(vRout[j]) == 1)[0]:
            G.add_edge("Input "+str(j),i,weight=1)
            actEdges.append(("Input "+str(j),i))
        for i in np.where(np.asarray(vRout[j]) == -1)[0]:
            G.add_edge("Input "+str(j),i,weight=-1)
            inhEdges.append(("Input "+str(j),i))
    
    node_opts = {"node_color": "white",
                 "edgecolors": "black",
                 "node_size": 800,
                 "linewidths": 2,
                 }
    nx.draw_networkx_nodes(G, posFullNet, **node_opts)

    if abs(adj_matrix).sum() != 0:
        for i in range(nw_dim):
            for ii in range(nw_dim):
                try:
                    if G.get_edge_data(i,ii)["weight"] == 1:
                        actEdges.append((i,ii))
                    elif G.get_edge_data(i,ii)["weight"] == -1:
                        inhEdges.append((i,ii))
                except:
                    pass
        
    nx.draw_networkx_edges(
        G,
        posFullNet,
        edgelist=actEdges,
        edge_color="k",
        arrowsize=20,
        width=2,    
    )
    nx.draw_networkx_edges(
        G,
        posFullNet,
        edgelist=inhEdges,
        alpha=0.8,
        edge_color="tab:red",
        arrowsize=20, 
        width=2
    )

    labels = {}
    for i in range(nw_dim):
        labels[i] = "X"+str(i)

    labelInput = {}
    for j in range(len(vRout)):
        labelInput["Input "+str(j)] = "Input "+str(j)
        
    nx.draw_networkx_labels(G, posFullNet, labels, font_size=14)
    nx.draw_networkx_labels(G, pos, labelInput, font_size=9)
    plt.tight_layout()
    plt.axis("off")
  
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
                dW=sigma*np.sqrt(dt)*np.random.normal() # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
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
                dW=sigma*np.sqrt(dt)*np.random.normal() # Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
                x[i,:] = x_next + dW
            
        return np.vstack((t,x.T))