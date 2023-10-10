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


import re

with open("230823SNIC_ToyMod.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []
inCm = 1/2.54


for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat = np.asarray(data)

plt.figure(figsize=(8.6*inCm,6*inCm))

idHB=21
end_us=28
plt.plot(dat[:idHB,3],dat[:idHB,6],'-b')
plt.plot(dat[idHB-1:end_us,3],dat[idHB-1:end_us,6],':k')

id_orb1 = 29
id_orb1_end = 50
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,6],'-g')
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,9],'-g')

id_SN = 66
id_SN_end = 77
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 86
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

id_SN = 87
id_SN_end = 99
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 107
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

id_SN = 107
id_SN_end = 119
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 127
plt.plot(dat[id_SN_end-1:id_us_end,3],dat[id_SN_end-1:id_us_end,6],':k')

id_SN = 127
id_SN_end = 139
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 146
plt.plot(dat[id_SN_end-1:id_us_end,3],dat[id_SN_end-1:id_us_end,6],':k')

plt.ylim(0,2)
plt.yticks([0,0.5,1,1.5,2], fontsize=8)
plt.xlim(-0.1,1)
plt.xticks(fontsize=8)
plt.xlabel('$r$',fontsize=10)
plt.ylabel('$x$',fontsize=10)
plt.tight_layout()