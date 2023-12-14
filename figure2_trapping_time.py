#!/usr/bin/env python
# coding: utf-8

# ## initialisation

# In[6]:


import numpy as np
from numpy.linalg import eig
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import *
from matplotlib.patches import Rectangle
import sympy as sp
from scipy.ndimage.morphology import binary_dilation
import math as math
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

# from numpy import *

params = {'legend.fontsize': 15,
          'axes.labelsize': 20,
          'axes.labelpad' : 15,
          'axes.titlesize':20,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
         'lines.markersize':4}
pylab.rcParams.update(params)




def sn_model(states):
    
    # alpha= 0.01
    
    x1   = states[:,0]
    x2   = states[:,1] 
    
    dx = np.zeros(states.shape) 
    
    dx[:,0] = alpha+x1**2
    dx[:,1] = -x2
    
    return dx


def q(states,diff): 
    

    dx    = diff(states)
    
    return  ((dx[:,0])**2 + (dx[:,1])**2)*0.5



def jac_sn(x1,x2):
    jacobian=np.array([[2*x1 ,0],
                 [0     ,-1]])

    return jacobian 

# In[35]:


def tan_inv(x,xinit,xfin):    #for ghost
    y =(1/(np.sqrt(0.01)))*(np.arctan((xfin)/np.sqrt(0.01))-np.arctan((xinit)/np.sqrt(0.01))) 
    return y
def log_inv(x,xinit,xfin):    #for saddle
    y =(1/(2*np.sqrt(0.4)))*(np.log(np.abs(xfin-np.sqrt(0.4))/(xfin + np.sqrt(0.4)))-np.log(np.abs(xinit-np.sqrt(0.4))/(xinit + np.sqrt(0.4)))) 
    return y

#%%
# ## function initialisation

# alpha=-0.4 # saddle    
alpha=0.01 # ghost

if alpha>0:
    xmin=-0.5;xmax=0.5
    ymin=-0.5;ymax=0.5
elif alpha<0:
    # xmin=0.2;xmax=0.7
    # ymin=-0.2;ymax=0.2
    xmin=0.13;xmax=1.13
    ymin=-0.5;ymax=0.5


# xmin=-0.5
# xmax=0.5
# ymin=-0.5
# ymax=0.5

diff=sn_model
q_thresh=0.01
npxl=101
n=101

x1=np.linspace(xmin,xmax,npxl)
x2=np.linspace(ymin,ymax,npxl)
grid_ss = np.meshgrid(x1, x2)
Xg,Yg=grid_ss


q_plot=np.zeros((npxl,npxl))
eig_plot=np.zeros((npxl,npxl))
U=np.zeros((len(x1),len(x2)));V=np.zeros((len(x1),len(x2)))

            
for j in range(len(x2)):
    for i in range(len(x1)):   
        x=np.array([[x1[i],x2[j]]])
        q_plot[j,i]=q(x,diff)
        if q_plot[j,i]<q_thresh:             
            jacobian = jac_sn(x1[i],x2[j])            #change jacobian for different models
            eigen_values =max(np.real(eig(jacobian)[0]))
            eig_plot[j,i]=eigen_values
        else:
            eig_plot[j,i]=np.nan
        dx=diff(x)
        U[j,i],V[j,i]=dx[:,0],dx[:,1]       
        
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(q_plot,origin='lower',extent=[xmin,xmax,ymin,ymax],cmap='coolwarm',norm=colors.LogNorm())
plt.streamplot(x1,x2,U,V,density=1,color=[0.1,0.1,0.1,0.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
plt.imshow(eig_plot,origin='lower',extent=[xmin,xmax,ymin,ymax],cmap='coolwarm')
plt.colorbar()
plt.streamplot(x1,x2,U,V,density=2,color=[0.1,0.1,0.1,0.5])      


# In[46]:


N             = 1
h             = 0.01
ran           = 50000
state_tf      = []

for j in range(N):
    states        = np.array([[-0.5,0.2]])
    state_tf_i   = []
    
    for i in range(ran):
    
            k1 = h*diff(states)
            k2 = h*diff(states+k1/2)
            k3 = h*diff(states+k2/2)
            k4 = h*diff(states+k3)
            states = states + (k1 + 2*k2 + 2*k3 + k4)/6  #+ 0.0001*np.sqrt(h)*np.random.normal((1,2))   

            if q(states,diff)<q_thresh: 
                state_tf_i.append(states)
                    
    state_tf.append(state_tf_i)

#


# In[50]:


t_trap_final2  =[]
eig_list_final2=[]

jac=jac_sn
width         = 0.01               #grid length
init_list     =[]
t_trap_theory=[]
eig_trap_theory=[]

length=0
for i in range(N):

    t_trap_i   =  []
    eig_list_i =  []
    
    t_trap_theory_=[]
    eig_trap_theory_=[]
    
    state_tf_i = np.array(state_tf[i]).reshape((len(state_tf[i]),2))
    

    for k in range(0,len(state_tf_i),20):

            if q((state_tf_i[k,:]).reshape(1,2),diff)<q_thresh:

                count=[]
                init_store=[]
                
                for j in range(len(state_tf_i)):
                                      
                    if state_tf_i[j,0]<state_tf_i[k,0]+width:
                        if state_tf_i[j,0]>state_tf_i[k,0]-width:
                             if state_tf_i[j,1]<state_tf_i[k,1]+width:
                                    if state_tf_i[j,1]>state_tf_i[k,1]-width:
                                              count.append(1)   
                                              init_store.append(state_tf_i[j])

                init_list.append(np.array([init_store[0],init_store[-1]]))
                
                
                jacobian = jac(state_tf_i[k,0],state_tf_i[k,1])            #change jacobian for different models
                eig_value =max(np.real(np.linalg.eig(jacobian)[0]))
                     
                eig_list_i.append(eig_value)
                t_trap_i.append(len(count)*0.01)
                length=length+1
                xinit=init_store[0][0]
                xfin=init_store[-1][0]
                
                

    if t_trap_i!=[] and eig_list_i!=[]:
        t_trap_final2.append(t_trap_i)
        eig_list_final2.append(eig_list_i)
        t_trap_theory.append(tan_inv(eig_value,xinit,xfin))
        eig_trap_theory.append((xinit+xfin))


# In[51]:


init_list=np.array(init_list)
t_trap_model3=[]
eig_trap_model3=[]
t_trap_model_var3=[]

eig_min=np.nanmin(np.nanmin(eig_list_final2))
eig_max=np.nanmax(np.nanmax(eig_list_final2))

t_trap_theory=[]
eig_trap_theory=[]

init_spacing=0
spacing=0.005

eig_grid=np.arange(eig_min,eig_max,spacing)

kini=eig_grid[0]

for k in eig_grid:
     
    eig_=[]
    teval=[]
    t_trap_theory_=[]
    eig_trap_theory_=[]
    
    for i in range(len(eig_list_final2)):
        for j in range(len(eig_list_final2[i])):
            if eig_list_final2[i][j]<k and eig_list_final2[i][j]>kini:
                eig_.append(eig_list_final2[i][j])
                teval.append(t_trap_final2[i][j])
                xinit=init_list[j][0][0]
                xfin=init_list[j][1][0]
                
                t_trap_theory_.append(log_inv(eig_list_final2[i][j],xinit,xfin))
                eig_trap_theory_.append((xinit+xfin))
#                 trapping_time_var.append(np.var(teval))

                eig_trap_theory_.append(eig_list_final2[i][j])

    kini=k
    if teval!=[] or eig_!=[]:

        t_trap_model3.append(np.mean(teval))
        eig_trap_model3.append(np.mean(eig_))
        t_trap_model_var3.append(np.std(teval))
        t_trap_theory.append(np.mean(t_trap_theory_))
        eig_trap_theory.append(np.mean(eig_trap_theory_))


# In[54]:


if alpha>0:
    plt.figure()
    plt.plot(eig_trap_model3,t_trap_model3,t_trap_model_var3,color='k',marker='s',ls='None',markersize=5,label='numerical')
    # plt.plot(eig_trap_model3,t_trap_theory,color='r',label='analytical')
    plt.xlabel(r'$\lambda_{g}^s$')
    plt.ylabel(r'piecewise trapping time $(\tau_i)$')
    plt.ylim(0,2.1)
    plt.xlim(-0.6,0.6)
    # plt.legend()
    plt.axvline(x=0,ls='--',color='gray')
    plt.show()

if alpha<0:
    xsaddle=2*np.sqrt(abs(alpha))
    
    folder_load1=folder_load+'new_results saddle\\'
    
    sn_analytic_eig=np.loadtxt(os.path.join(folder_load1,'sn_saddle_%.1f_analytic_eig.npy'%abs(alpha)))
    sn_analytic_trapping=np.loadtxt(os.path.join(folder_load1,'sn_saddle_%.1f_analytic_trapping.npy'%abs(alpha)))
    sn_numeric_eig=np.loadtxt(os.path.join(folder_load1,'sn_saddle_%.1f_numeric_eig.npy'%abs(alpha)))
    sn_numeric_trapping=np.loadtxt(os.path.join(folder_load1,'sn_saddle_%.1f_numeric_trapping.npy'%abs(alpha)))
    
    plt.figure()
    plt.plot(sn_numeric_eig,sn_numeric_trapping,color='k',marker='s',ls='None',markersize=5,label='numerical')
    # plt.plot(sn_analytic_eig,sn_analytic_trapping,color='r',label='analytical')
    plt.xlabel(r'$\lambda_{max}^s$')
    plt.ylabel(r'piecewise trapping time $(\tau_i)$')
    plt.ylim(0,1.5)
    plt.xlim(xsaddle-0.01,1.435)
    # plt.legend()
    plt.axvline(x=xsaddle,ls='--',color='gray')
    plt.show()



