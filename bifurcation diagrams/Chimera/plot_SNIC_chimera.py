# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:27:20 2023

@author: koch
"""
# %matplotlib qt 

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

inCm = 1/2.54

with open("230823_chimera_lowerHB.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

datHB_low = np.asarray(data)

with open("230823_chimera_upperHB.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

datHB_up = np.asarray(data)

with open("230823_chimeraSN.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

datSN = np.asarray(data)


#plotting

plt.figure(figsize=(4.5*inCm,4*inCm))

#plot lower HB
idHB=20
end_us=28
plt.plot(datHB_low[:idHB,3],datHB_low[:idHB,6],'-b')
plt.plot(datHB_low[idHB-1:end_us,3],datHB_low[idHB-1:end_us,6],'--k')

id_orb1 = 28
id_orb1_end = 64
plt.plot(datHB_low[id_orb1:id_orb1_end,3],datHB_low[id_orb1:id_orb1_end,6],'-g')
plt.plot(datHB_low[id_orb1:id_orb1_end,3],datHB_low[id_orb1:id_orb1_end,9],'-g')

#plot upper HB
idHB=20
end_us=28
plt.plot(datHB_up[:idHB,3],datHB_up[:idHB,6],'-b')
plt.plot(datHB_up[idHB-1:end_us,3],datHB_up[idHB-1:end_us,6],'--k')

id_orb1 = 28
id_orb1_end = 64
plt.plot(datHB_up[id_orb1:id_orb1_end,3],datHB_up[id_orb1:id_orb1_end,6],'-g')
plt.plot(datHB_up[id_orb1:id_orb1_end,3],datHB_up[id_orb1:id_orb1_end,9],'-g')

#plot SNs

id_SN = 1
id_SN_end = 12
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-b')
id_us_end = 22
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 22
id_SN_end = 34
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-b')
id_us_end = 42
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 42
id_SN_end = 54
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-b')
id_us_end = 62
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 62
id_SN_end = 74
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-b')
id_us_end = 82
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 82
id_SN_end = 94
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-b')
id_us_end = 102
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')

id_SN = 102
id_SN_end = 114
plt.plot(datSN[id_SN:id_SN_end,3],datSN[id_SN:id_SN_end,6],'-b')
id_us_end = 229
plt.plot(datSN[id_SN_end:id_us_end,3],datSN[id_SN_end:id_us_end,6],'--k')


plt.ylim(0,3)
plt.yticks([0,1,2,3], fontsize=8)
plt.xlim(-0.1,0.6)
plt.xticks([0,0.25,0.5], fontsize=8)
plt.xticks(fontsize=8)
plt.xlabel('$\\alpha$',fontsize=10)
plt.ylabel('$x$',fontsize=10)
plt.tight_layout()