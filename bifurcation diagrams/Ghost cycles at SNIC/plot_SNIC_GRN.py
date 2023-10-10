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

with open("230821SNIC_GRN.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat = np.asarray(data)

plt.figure(figsize=(8.6*inCm,6*inCm))

idHB=28
end_us=65
plt.plot(dat[:idHB,3],dat[:idHB,6],'-b')
plt.plot(dat[idHB-1:end_us,3],dat[idHB-1:end_us,6],':k')

id_orb1 = 65
id_orb1_end = 224
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,6],'-g')
plt.plot(dat[id_orb1:id_orb1_end,3],dat[id_orb1:id_orb1_end,9],'-g')

id_SN = 242
id_SN_end = 312
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 333
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')


id_SN = 333
id_SN_end = 403
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 422
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

id_SN = 422
id_SN_end = 493
plt.plot(dat[id_SN:id_SN_end,3],dat[id_SN:id_SN_end,6],'-b')
id_us_end = 511
plt.plot(dat[id_SN_end:id_us_end,3],dat[id_SN_end:id_us_end,6],':k')

# plt.plot(dat[:,3],dat[:,9])

plt.ylim(1e-3,11)
plt.gca().set_yticklabels([1e-3,1e-2,1e-1,1e0,10],fontsize=8)
plt.yscale('log')
plt.xlim(0,2)
plt.xticks([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00])
plt.gca().set_xticklabels([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00],fontsize=8)
plt.xlabel('$g$',fontsize=10)
plt.ylabel('$x_1$',fontsize=10)
plt.tight_layout()