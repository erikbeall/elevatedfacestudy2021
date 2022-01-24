#!/usr/bin/env python3

import numpy as np
import scipy
import csv
import glob
from scipy import signal
import copy
import matplotlib.pylab as plt
plt.ion()
np.set_printoptions(precision=3)

import datetime
import time

markersize=6
full_dataset = np.load('stolaf_full_dataset.npy')
'''
0 = Air_temperature_snapshot_room
1 = Oral Temperature
2 = Quick Oral Temperature
3 = Surface canthus temperature inside tent at end of 10-minutes
4 = delta time between end of 10 minutes and snapshot
    for using the last measured env temp and last equil image this must be <2 minutes
    np.sum(full_dataset[:,:,4]>120)/112 is 15% of the data
5 = effective air/environmental temperature in last 100 seconds of tent
6 = Surface canthus temperature outside tent during secondary scans
7 = Device #5 surface temperature
8 = Device #2 Body temperature
9 = Device #3 Body temperature
10 = Device #4 Scan Body temperature
11 = air temperature (not adjusted with background, just raw air) over last 100 seconds in tent
'''

plt.figure()
plt.subplot(1,2,1)
cdata=np.genfromtxt('cargill_data.csv', delimiter=',')
cdata=cdata[1:,:]
oral = cdata[:,1]
fi_75F = cdata[:,2]
fi_85F = cdata[:,6]
fi_85F[-1]=round(np.nanmean(fi_85F),1)

# product correction factor F/(1+F) used at time of pilot study
unphysio = lambda core, amb: core - 0.191*(core-amb)
surf_85F = unphysio(fi_85F, 85)
surf_75F = unphysio(fi_75F, 75)
surf_45F = np.loadtxt('cargill_45F.txt')

# ambient data not recorded beyond air probe at time of study
alldata_ambients = np.concatenate([np.ones(len(surf_75F))*75, np.ones(len(surf_85F))*85])
alldata_surfs = np.concatenate([surf_75F, surf_85F])

p=np.polyfit(alldata_ambients, alldata_surfs,1)
#plt.plot(np.linspace(70, 90, 100), np.polyval(p, np.linspace(70, 90, 100)))
plt.plot(np.ones(len(surf_45F))*45, surf_45F,'k.')
plt.plot(np.linspace(40, 90, 100), np.polyval(p, np.linspace(40, 90, 100)), 'k--')
plt.plot(np.ones(len(surf_75F))*75, surf_75F,'k.')
plt.plot(np.ones(len(surf_85F))*85, surf_85F,'k.')
plt.legend(['Surface Data', 'Fit to 75, 85F, F=0.234'])
plt.ylabel('Surface (F)', fontsize=12)
plt.xlabel('Ambient (F)', fontsize=12)

axa=plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
effectives=full_dataset[:,:,5].flatten()
airs=full_dataset[:,:,11].flatten()
surfs=full_dataset[:,:,3].flatten()
T_oral=(full_dataset[:,:,1].flatten()-32)/1.8
orals=T_oral.flatten()
offset=orals - surfs
orals=orals[np.isnan(airs)==False]
offset=offset[np.isnan(airs)==False]
effectives=effectives[np.isnan(airs)==False]
surfs=surfs[np.isnan(airs)==False]
airs=airs[np.isnan(airs)==False]
# fit against the effective air temperature
p=np.polyfit(effectives, offset, 1)
f=-p[0]
F = f/(1-f)
physio_correction = lambda surf, amb: surf + (surf-amb)*F
un_physio_correction = lambda core, amb: core - (core-amb)*f
print('Fit of F-value to elevated tent temperatures: ', F)

plt.plot(effectives*1.8+32, orals*1.8+32, 'b.', markersize=markersize)
plt.plot(effectives*1.8+32, surfs*1.8+32, 'ks', markersize=markersize)
plt.plot(effectives*1.8+32, physio_correction(surfs, effectives)*1.8+32, 'r.', markersize=markersize)
mn=np.min(effectives)*1.8+32
mx=np.max(effectives)*1.8+32
x=np.linspace(mn, mx, 20)
po=np.polyfit(effectives*1.8+32, orals*1.8+32, 1)
ps=np.polyfit(effectives*1.8+32, surfs*1.8+32, 1)
pb=np.polyfit(effectives*1.8+32, physio_correction(surfs, effectives)*1.8+32, 1)
y=np.polyval(po, x)
plt.plot(x, y, 'b--')
y=np.polyval(ps, x)
plt.plot(x, y, 'k--')
y=np.polyval(pb, x)
plt.plot(x, y, 'r--')
plt.xlabel('Ambient (F)', fontsize=12)
plt.ylabel('Measured (F)', fontsize=12)
plt.legend(['Oral Thermometry','Surface Temperature', 'Corrected Body, F=%.3f'%F])
plt.savefig('F_value_pilot_and_study.png', bbox_inches='tight',pad_inches = 0.1)

