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

# fever inspect updated factor from data collected under a Business Associate agreement with a medical facility
# NOTE to users: this factor likely incorporates some system-dependence and should be evaluated for any changes in
# the entire optical path, including optical coatings, image processing/calibration chain
# and all things related to pixel spot size and pixel dependence, which includes resolution, 
# FOV, blur and crosstalk, among others ... Be very wary of trusting results until the device can pass all tests.
F=0.1595
f=F/(1+F)
physio_correction = lambda surf, amb: surf + (surf-amb)*F
un_physio_correction = lambda core, amb: core - (core-amb)*f

markersize=20
# A simple quicker test requirement for development purposes (its not easy to line up 20 subjects and heated tents)
# is the ability to observe clean, reproducible equilibration curves, with the caveat there is
# variability from subject physiology. Nevertheless, one should be able to reproduce Fig 4 (equilibration curve)
# in several subjects at several different times and start/end temperatures. Getting that curve only one in ten tries
# indicates something is wrong, should be able to get it at least six or more out of ten randomly chosen attempts.
# only Device #1 and #5 were able to pass this test.
# linear fit for logarithm of temperatures over time (conditioned to bound at log(0.001))
linfit = lambda surfs, times: np.polyfit(times - np.min(times), np.log(0.001 + max(surfs) - surfs), 1)

demographics = np.load('stolaf_demographics.npy')
# contains height, weight, age, gender (1==male, 0==not male)

full_dataset = np.load('stolaf_full_dataset.npy')
'''
DATASET: stolaf_full_dataset.npy
full_dataset is shaped (28,4), for (subject,tent)
Note: tents C and D were swapped for first 11 datasets 
because C was the second-hottest tent for first 11, then it was the hottest tent
so now in this .npy, in every subject the order goes from coldest tent to hottest tent

columns are:
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

curves_fsurfs, curves_fenvironment, curves_fdistances, curves_fambient, curves_fradiative = np.load('stolaf_filtered_curve_data.npy')
'''
DATASET: stolaf_filtered_curve_data.npy
curves_X is shaped (28,4,600), for (subject,tent,seconds)
curves contain the linear interpolated (piecewise cubic with endpoints interpolated from the penultimate data)
 these were also filtered with a Savitsky-Golay filter
curves_fsurfs is the canthus temperatures
curves_fenvironment is the environmental effective temperature

# model T(t) = T_f - (T_f - T_o)*exp(-t/tau), where tau ~ 1/RC = 95 seconds plus or minus 20%
#          1.5 min,        3 min,      4.5 min,     6 min,       7.5 min,     9 min
#         1 tau = 37%, 2 tau = 14%, 3 tau = 5%, 4 tau = 1.8%, 5 tau = 0.7%, 6 tau = 0.2%
# air 68->78F   = 4F,         1.4F,       0.5F,        0.2F,         0.07F,     0.02F
# skin 93->94.8=0.673,       0.25F,      0.09F,        0.04F
# underest      = 0.9F,      0.36F,      0.18F,        0.11F

Below I've included a function to match the exponential relation over time for a changing environmental temperature
so if T_effectives is all one value, this collapses to the exponential form above
And if T_effectives is changing, as was found with the V1 study environments, 
you can use this to generate expected skin temperatures. However,
individual subjects appear to have had varying amounts of convective going on, which alters
the fenvironment curve above. The potential error must be kept in mind, and going forward, 
the environment must be modified in order to reduce convective and varying air to below 0.5C.
systematic error is estimated from the variability and the delta between methods
'''

# tau is an estimate based on fits to faces exposed to heated blown air and then normal room
# this is likely different from that experienced in realistic settings
def gen_T_equilibration(T_effectives, T_surf_initial, T_oral, tau=94.0):
    surfs=[T_surf_initial]
    for i,T in enumerate(T_effectives):
        surfs.append(surfs[-1]+(un_physio_correction(T_oral, T)-surfs[-1])/tau)
    return np.array(surfs[:-1])


all_uuids=np.load('uuid_list.npy')
# supplement detailed tables of air vs effective temperature, each device's body temperatures (except surface temps for device 5)
table_efftemp_tent=full_dataset[:,:,5]
table_airtemp_tent=full_dataset[:,:,11]
table_radtemp_tent=np.mean(curves_fradiative[:,:,-100:],2)
fp=open('tent_temperatures.csv','w')
fp.write('Subject,TentA-Air,TentB-Air,TentC-Air,TentD-Air,TentA-Rad,TentB-Rad,TentC-Rad,TentD-Rad,TentA-Eff,TentB-Eff,TentC-Eff,TentD-Eff\n')
for i,(row_a, row_r, row_e) in enumerate(zip(table_airtemp_tent, table_radtemp_tent, table_efftemp_tent)):
    fp.write('%s,'%(all_uuids[i]))
    fp.write('%.2f,%.2f,%.2f,%.2f,'%tuple(row_a*1.8+32))
    fp.write('%.2f,%.2f,%.2f,%.2f,'%tuple(row_r*1.8+32))
    fp.write('%.2f,%.2f,%.2f,%.2f\n'%tuple(row_e*1.8+32))
fp.close()

table_device1= physio_correction(full_dataset[:,:,6], full_dataset[:,:,0])
table_device2=full_dataset[:,:,8]
table_device3=full_dataset[:,:,9]
table_device4=full_dataset[:,:,10]
table_device5=full_dataset[:,:,7]
# convert C to F
table_device1=table_device1*1.8 + 32
table_device5=table_device5*1.8 + 32
table_devices=[table_device1, table_device2, table_device3, table_device4, table_device5]
for i in range(5):
    fp=open('device_temps%d.csv'%(i+1),'w')
    fp.write('Subject,TentA,TentB,TentC,TentD\n')
    for j,row in enumerate(table_devices[i]):
        fp.write('%s,'%(all_uuids[j]))
        fp.write('%.2f,%.2f,%.2f,%.2f\n'%tuple(row))
    fp.close()

fp=open('device_temps_tentD.csv','w')
fp.write('Subject,Device 1,Device 2,Device 3,Device 4,Device 5\n')
for i,row in enumerate(table_devices[0]):
    fp.write('%s,'%(all_uuids[i]))
    fp.write('%.2f,%.2f,%.2f,%.2f,%.2f\n'%tuple([table_device1[i,-1], table_device2[i,-1],table_device3[i,-1],table_device4[i,-1],table_device5[i,-1]]))
fp.close()


# first, get the oral and the quick oral temps
T_oral=(full_dataset[:,:,1].flatten()-32)/1.8
T_quickoral=(full_dataset[:,:,2].flatten()-32)/1.8

fp=open('oral_temps.csv','w')
fp.write('Subject,OralQA,OralQB,OralQC,OralQD,OralA,OralB,OralC,OralD\n')
for i,(row_o, row_q) in enumerate(zip(T_oral.reshape((28,4)), T_quickoral.reshape((28,4)))):
    fp.write('%s,'%(all_uuids[i]))
    fp.write('%.2f,%.2f,%.2f,%.2f,'%tuple(row_q*1.8+32))
    fp.write('%.2f,%.2f,%.2f,%.2f\n'%tuple(row_o*1.8+32))

fp.close()

# F-value fitting: oral-surface vs elevated temperature
effectives=full_dataset[:,:,5].flatten()
airs=full_dataset[:,:,11].flatten()
surfs=full_dataset[:,:,3].flatten()
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


fp=open('tents_snapshot_airtemps.csv','w')
fp.write('Subject,Ambient A, Ambient B, Ambient C, Ambient D\n')
for i,row in enumerate(full_dataset[:,:,0]):
    fp.write('%s,'%(all_uuids[i]))
    fp.write('%.2f,%.2f,%.2f,%.2f\n'%tuple(row))
fp.close()


# first, get the oral and the quick oral temps
T_oral=(full_dataset[:,:,1].flatten()-32)/1.8
T_quickoral=(full_dataset[:,:,2].flatten()-32)/1.8


# generate the elevated body temperatures from the final surface temperature recorded inside the tent
T_simulated_fromsurf = physio_correction(full_dataset[:,:,3], full_dataset[:,:,0])
T_simulated_fromsurf[T_simulated_fromsurf<10] = np.nan
print('Elevated-from-Surface temps: ', np.nanmean(T_simulated_fromsurf, 0)*1.8+32, ', min/max=', np.nanmin(T_simulated_fromsurf)*1.8+32, np.nanmax(T_simulated_fromsurf)*1.8+32, ', stdev: ', np.nanstd(T_simulated_fromsurf, 0)*1.8)
# generate the expected elevated temperatures from the orals, and
# the environment temperature inside the tent and outside the tent
T_simulated_fromoral = physio_correction(un_physio_correction(T_oral.reshape((28,4)), full_dataset[:,:,5]), full_dataset[:,:,0])
T_simulated_fromoral[T_simulated_fromoral<10] = np.nan
print('Elevated-from-Oral temps: ', np.nanmean(T_simulated_fromoral, 0)*1.8+32, ', min/max=', np.nanmin(T_simulated_fromoral)*1.8+32, np.nanmax(T_simulated_fromoral)*1.8+32, ', stdev: ', np.nanstd(T_simulated_fromoral, 0)*1.8)
# there are two issues with the alternative method:
# 1) it is more coupled to the fever inspect device's actual accuracy (and thus less fair to Devices 2-5 as a target)
# 2) it is not derived from oral temperatures, and we find it has greater variability than the oral temperatures
# and produces some temperatures lower than the lowest oral (by 0.6F). Conversely, the first method is dependent
# on the accuracies/stabilities of two air temperatures and doubly dependent on the accuracy/variability 
# of the physiologic factor vs some ideal, actual physiologic factor F. 
# Ultimately, we chose to display both methods, with the second method reserved or relegated to the Supplement. 
# Nevertheless, the final analysis shows the alternative method produces better fidelity to T_measured 
# across all devices.

fp=open('elevated_temps_surfbased.csv','w')
fp.write('Subject,Tent A, Tent B, Tent C, Tent D\n')
for i,row in enumerate(T_simulated_fromsurf*1.8+32):
    fp.write('%s,'%(all_uuids[i]))
    fp.write('%.2f,%.2f,%.2f,%.2f\n'%tuple(row))
fp.close()
fp=open('elevated_temps_oralbased.csv','w')
fp.write('Subject,Tent A, Tent B, Tent C, Tent D\n')
for i,row in enumerate(T_simulated_fromoral*1.8+32):
    fp.write('%s,'%(all_uuids[i]))
    fp.write('%.2f,%.2f,%.2f,%.2f\n'%tuple(row))
fp.close()

plt.close()
plt.figure()
#
# histogram oral temperatures - optional
#plt.subplot(1,2,1)
#_=plt.hist(T_oral.flatten()*1.8+32,12);
#_=plt.hist(np.nanmean(T_oral,1)*1.8+32,12);
#plt.xlabel('Oral (F)')
#
# squish the plots vertically so they look more compact
# by making it a 4x4 grid - with bbox_inches tight and no padding, 
# matplotlib will simply save the non-empty frames
plt.subplot(2,2,1)
airs=full_dataset[:,:,5].flatten()
# effective air temperature
airs[airs==0]=np.nan
airs=airs[airs>0]
_=plt.hist(airs*1.8+32,12);
plt.xlabel('Effective Air (F)', fontsize=14)
plt.subplot(2,2,2)
elevs=np.copy(T_simulated_fromsurf)
elevs[elevs==0]=np.nan
elevs=elevs[elevs>0]
_=plt.hist(elevs*1.8+32,12);
plt.xlabel('Elevated Body (F)', fontsize=14)
plt.savefig('Fig_elevated.png', bbox_inches='tight',pad_inches = 0.1)

plt.close()
plt.figure()
plt.subplot(2,2,1)
elevs=np.copy(T_simulated_fromsurf)
elevs[elevs==0]=np.nan
elevs=elevs[elevs>0]
_=plt.hist(elevs*1.8+32,12);
plt.xlabel('Surface-based Body (F)', fontsize=14)
plt.subplot(2,2,2)
elevs=np.copy(T_simulated_fromoral)
elevs[elevs==0]=np.nan
elevs=elevs[elevs>0]
_=plt.hist(elevs*1.8+32,12);
plt.xlabel('Oral-based Body (F)', fontsize=14)
plt.savefig('SFig_surfbased_elevated.png', bbox_inches='tight',pad_inches = 0.1)

# make the bias-to-normal plots, first making a useful grid (x-axis) for the oral-based simulated body temps
x=T_simulated_fromoral.flatten()*1.8+32
inds=np.isnan(x)==False
x=x[inds]
x_orig=np.copy(x)
xs=np.linspace(np.nanmin(x), np.nanmax(x), 100)

# and a grid for the surface temp-based T_simulated_fromsurf
ax=T_simulated_fromsurf.flatten()*1.8+32
# inds is the same for both alt and oral temp-based simulated body temps
ax=ax[inds]
ax_orig=np.copy(ax)
axs=np.linspace(np.nanmin(ax), np.nanmax(ax), 100)

# internal device thresholds
# these are used instead of 100.4F across-the-board, to be as fair as possible to the devices
thr1=100.4
thr2=99.14
thr3=100.1
thr4=100.4
# the surface mode-only device requires a different treatment tailored to its operating mode
# Threshold arrived at via Eqns in paper with ambient air temperature of 
# 21.78C (the average room temperature), then adding 1C
# Device #5's method is the average of last 8 subjects' skin temps (average of our orals is just below 37C)
# and then add 1C after this is calculated. This method hews as closely to that as possible while using
# the average subject body temperature (orals, 36.864C) to be as fair as possible
thr5= (un_physio_correction(36.864, 21.78)+1)*1.8+32
## thr5=96.42
# so this becomes the threshold used for selecting elevated temperatures as true or false febrile for device 5
# and to determine whether device 5 has detected a temperature as elevated or not, we either cannot assess this OR
thr5_selection=(36.864+1)*1.8+32

# get the device body temps
# Device #1, Device #2, Device #3, Device #4, Device #5
y=physio_correction(full_dataset[:,:,6].flatten(), full_dataset[:,:,0].flatten())*1.8+32
y=y[inds]
this_inds=np.isnan(y)==False
y=y[this_inds]
y_device1=np.copy(y)
x1=x_orig[this_inds]
ax1=ax_orig[this_inds]

y=full_dataset[:,:,8].flatten()
y=y[inds]
this_inds=np.isnan(y)==False
y=y[this_inds]
y_device2=np.copy(y)
x2=x_orig[this_inds]
ax2=ax_orig[this_inds]

y=full_dataset[:,:,9].flatten()
y=y[inds]
this_inds=np.isnan(y)==False
y=y[this_inds]
y_device3=np.copy(y)
x3=x_orig[this_inds]
ax3=ax_orig[this_inds]

y=full_dataset[:,:,10].flatten()
y=y[inds]
this_inds=np.isnan(y)==False
y=y[this_inds]
y_device4=np.copy(y)
x4=x_orig[this_inds]
ax4=ax_orig[this_inds]

# surface mode only for Device #5 - leads to confusion in interpreting its plots vs the other devices
y=full_dataset[:,:,7].flatten()*1.8+32
y=y[inds]
this_inds=np.isnan(y)==False
y=y[this_inds]
# get the air temperatures in test room during each measurement
airtemps = full_dataset[:,:,0].flatten()
airtemps = airtemps[inds]
airtemps_device5 = np.copy(airtemps[this_inds])
y_device5=np.copy(y)
x5=x_orig[this_inds]
ax5=ax_orig[this_inds]


# x-axis -> simulated elevated temperatures, aligned to the same indices
# get fits and residual errors
p1=np.polyfit(x1, y_device1, 1)
rsq1=1-np.var(y_device1-np.polyval(p1,x1))/np.var(y_device1)
ap1=np.polyfit(ax1, y_device1, 1)
arsq1=1-np.var(y_device1-np.polyval(ap1,ax1))/np.var(y_device1)

p2=np.polyfit(x2, y_device2, 1)
rsq2=1-np.var(y_device2-np.polyval(p2,x2))/np.var(y_device2)
ap2=np.polyfit(ax2, y_device2, 1)
arsq2=1-np.var(y_device2-np.polyval(ap2,ax2))/np.var(y_device2)

p3=np.polyfit(x3, y_device3, 1)
rsq3=1-np.var(y_device3-np.polyval(p3,x3))/np.var(y_device3)
ap3=np.polyfit(ax3, y_device3, 1)
arsq3=1-np.var(y_device3-np.polyval(ap3,ax3))/np.var(y_device3)

p4=np.polyfit(x4, y_device4, 1)
rsq4=1-np.var(y_device4-np.polyval(p4,x4))/np.var(y_device4)
ap4=np.polyfit(ax4, y_device4, 1)
arsq4=1-np.var(y_device4-np.polyval(ap4,ax4))/np.var(y_device4)

p5=np.polyfit(x5, y_device5, 1)
rsq5=1-np.var(y_device5-np.polyval(p5,x5))/np.var(y_device5)
ap5=np.polyfit(ax5, y_device5, 1)
arsq5=1-np.var(y_device5-np.polyval(ap5,ax5))/np.var(y_device5)
'''
y_device5_bt = [physio_correction(y,a*1.8+32) for y,a in zip(y_device5, airtemps_device5)]
p5_bt=np.polyfit(x5, y_device5_bt, 1)
rsq5_bt=1-np.var(y_device5_bt-np.polyval(p5,x5))/np.var(y_device5_bt)
ap5_bt=np.polyfit(ax5, y_device5_bt, 1)
arsq5_bt=1-np.var(y_device5_bt-np.polyval(ap5,ax5))/np.var(y_device5_bt)
'''


# plot the alternative (surface-based) T_simulated vs the alternative body (oral-based, x-axis) T_simulated
plt.close()
plt.figure()
plt.plot(x1, ax1, '.')
altp=np.polyfit(x1, ax1, 1)
# plot the trendline and a 1:1 line to show the bottom-line difference: the alternative method produces ~17% closer-to-normal simulated temps
plt.plot(np.linspace(97.5, 102, 100), np.polyval(altp, np.linspace(97.5, 102, 100)), '--')
plt.plot(np.linspace(97.5, 102, 100), np.linspace(97.5, 102, 100), 'k--')
plt.legend(['Surface- vs Oral-based Elevated', 'Surface= %.2f*Oral + %.2f'%(altp[0], altp[1]), '1:1 mapping'])
plt.savefig('SFig_surf_vs_oral_based_temps_fit.png', bbox_inches='tight',pad_inches = 0.1)


# make the main plots
# a) detected vs elevated straight line, compare with body-to-body (1:1)
# b) Bland-Altman
plt.close()
plt.figure()
plt.subplot(1,2,1)
plt.plot(x1, y_device1, '.')
plt.plot(xs, np.polyval(p1,xs), '--')
plt.plot(xs, xs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #1 vs Body','slope=%.2f, r$^2$=%.2f'%(p1[0], rsq1), 'Body vs Body'], loc=2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d1=y_device1 - x1
m1=np.nanmean(d1)
s1=2*np.nanstd(d1)
plt.plot(x1, d1, 'b.')
plt.plot(np.linspace(np.min(x1), np.max(x1), 20), np.ones((20,))*m1, 'k-.')
plt.plot(np.linspace(np.min(x1), np.max(x1), 20), np.ones((20,))*s1+m1, 'k--')
plt.plot(np.linspace(np.min(x1), np.max(x1), 20), -np.ones((20,))*s1+m1, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m1, '2$\sigma$ = $\pm$%.1fF'%s1])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_oralDevice #1.png', bbox_inches='tight',pad_inches = 0.1)

# get residuals of the BA detected minus elevated when fitted to a trendline vs flat


plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(ax1, y_device1, '.')
plt.plot(axs, np.polyval(ap1,axs), '--')
plt.plot(axs, axs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #1 vs Body','slope=%.2f, r$^2$=%.2f'%(ap1[0], arsq1), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d1=y_device1 - ax1
m1=np.nanmean(d1)
s1=2*np.nanstd(d1)
plt.plot(ax1, d1, 'b.')
plt.plot(np.linspace(np.min(ax1), np.max(ax1), 20), np.ones((20,))*m1, 'k-.')
plt.plot(np.linspace(np.min(ax1), np.max(ax1), 20), np.ones((20,))*s1+m1, 'k--')
plt.plot(np.linspace(np.min(ax1), np.max(ax1), 20), -np.ones((20,))*s1+m1, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m1, '2$\sigma$ = $\pm$%.1fF'%s1])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_surfDevice #1.png', bbox_inches='tight',pad_inches = 0.1)


all_fits = {}
all_fits['device1'] = [p1, rsq1, ap1, arsq1]

plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(x2, y_device2, '.')
plt.plot(xs, np.polyval(p2,xs), '--')
plt.plot(xs, xs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #2 vs Body','slope=%.2f, r$^2$=%.2f'%(p2[0], rsq2), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d2=y_device2 - x2
m2=np.nanmean(d2)
s2=2*np.nanstd(d2)
plt.plot(x2, d2, 'b.')
plt.plot(np.linspace(np.min(x2), np.max(x2), 20), np.ones((20,))*m2, 'k-.')
plt.plot(np.linspace(np.min(x2), np.max(x2), 20), np.ones((20,))*s2+m2, 'k--')
plt.plot(np.linspace(np.min(x2), np.max(x2), 20), -np.ones((20,))*s2+m2, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m2, '2$\sigma$ = $\pm$%.1fF'%s2])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_oralDevice #2.png', bbox_inches='tight',pad_inches = 0.1)

plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(ax2, y_device2, '.')
plt.plot(axs, np.polyval(ap2,axs), '--')
plt.plot(axs, axs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #2 vs Body','slope=%.2f, r$^2$=%.2f'%(ap2[0], arsq2), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d2=y_device2 - ax2
m2=np.nanmean(d2)
s2=2*np.nanstd(d2)
plt.plot(ax2, d2, 'b.')
plt.plot(np.linspace(np.min(ax2), np.max(ax2), 20), np.ones((20,))*m2, 'k-.')
plt.plot(np.linspace(np.min(ax2), np.max(ax2), 20), np.ones((20,))*s2+m2, 'k--')
plt.plot(np.linspace(np.min(ax2), np.max(ax2), 20), -np.ones((20,))*s2+m2, 'k--')
saved_ylim=plt.ylim()
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m2, '2$\sigma$ = $\pm$%.1fF'%s2])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_surfDevice #2.png', bbox_inches='tight',pad_inches = 0.1)

# SHOW EFFECT OF Bland-Altman that uses average as x-axis instead of the reference method
plt.close()
plt.figure()
axa = plt.subplot(1,2,1)
axa.yaxis.set_label_position("left")
axa.yaxis.tick_right()
max2=0.5*ax2 + 0.5*y_device2
d2=y_device2 - ax2
m2=np.nanmean(d2)
s2=2*np.nanstd(d2)
# Canonical Bland-Altman: use the average as the x-axis
plt.plot(max2, d2, 'b.')
plt.plot(np.linspace(np.min(max2), np.max(max2), 20), np.ones((20,))*m2, 'k-.')
plt.plot(np.linspace(np.min(max2), np.max(max2), 20), np.ones((20,))*s2+m2, 'k--')
plt.plot(np.linspace(np.min(max2), np.max(max2), 20), -np.ones((20,))*s2+m2, 'k--')
plt.ylim(saved_ylim)
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m2, '2$\sigma$ = $\pm$%.1fF'%s2])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Average (F)', fontsize=14)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d2=y_device2 - ax2
m2=np.nanmean(d2)
s2=2*np.nanstd(d2)
plt.plot(ax2, d2, 'b.')
plt.plot(np.linspace(np.min(ax2), np.max(ax2), 20), np.ones((20,))*m2, 'k-.')
plt.plot(np.linspace(np.min(ax2), np.max(ax2), 20), np.ones((20,))*s2+m2, 'k--')
plt.plot(np.linspace(np.min(ax2), np.max(ax2), 20), -np.ones((20,))*s2+m2, 'k--')
plt.ylim(saved_ylim)
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m2, '2$\sigma$ = $\pm$%.1fF'%s2])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Body (F)', fontsize=14)
plt.savefig('Method_BA_gold_vs_avg.png', bbox_inches='tight',pad_inches = 0.1)
all_fits['device2'] = [p2, rsq2, ap2, arsq2]


plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(x3, y_device3, '.')
xs=np.linspace(np.nanmin(x), np.nanmax(x), 100)
plt.plot(xs, np.polyval(p3,xs), '--')
print('slope again for device 3: ', p3)
plt.plot(xs, xs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #3 vs Body','slope=%.2f, r$^2$=%.2f'%(p3[0], rsq3), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d3=y_device3 - x3
m3=np.nanmean(d3)
s3=2*np.nanstd(d3)
plt.plot(x3, d3, 'b.')
plt.plot(np.linspace(np.min(x3), np.max(x3), 20), np.ones((20,))*m3, 'k-.')
plt.plot(np.linspace(np.min(x3), np.max(x3), 20), np.ones((20,))*s3+m3, 'k--')
plt.plot(np.linspace(np.min(x3), np.max(x3), 20), -np.ones((20,))*s3+m3, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m3, '2$\sigma$ = $\pm$%.1fF'%s3])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_oralDevice #3.png', bbox_inches='tight',pad_inches = 0.1)

plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(ax3, y_device3, '.')
plt.plot(axs, np.polyval(ap3,axs), '--')
plt.plot(axs, axs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #3 vs Body','slope=%.2f, r$^2$=%.2f'%(ap3[0], arsq3), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d3=y_device3 - ax3
m3=np.nanmean(d3)
s3=2*np.nanstd(d3)
plt.plot(ax3, d3, 'b.')
plt.plot(np.linspace(np.min(ax3), np.max(ax3), 20), np.ones((20,))*m3, 'k-.')
plt.plot(np.linspace(np.min(ax3), np.max(ax3), 20), np.ones((20,))*s3+m3, 'k--')
plt.plot(np.linspace(np.min(ax3), np.max(ax3), 20), -np.ones((20,))*s3+m3, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m3, '2$\sigma$ = $\pm$%.1fF'%s3])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_surfDevice #3.png', bbox_inches='tight',pad_inches = 0.1)

all_fits['device3'] = [p3, rsq3, ap3, arsq3]

plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(x4, y_device4, '.')
xs=np.linspace(np.nanmin(x), np.nanmax(x), 100)
plt.plot(xs, np.polyval(p4,xs), '--')
plt.plot(xs, xs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #4 vs Body','slope=%.2f, r$^2$=%.2f'%(p4[0], rsq4), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d4=y_device4 - x4
m4=np.nanmean(d4)
s4=2*np.nanstd(d4)
plt.plot(x4, d4, 'b.')
plt.plot(np.linspace(np.min(x4), np.max(x4), 20), np.ones((20,))*m4, 'k-.')
plt.plot(np.linspace(np.min(x4), np.max(x4), 20), np.ones((20,))*s4+m4, 'k--')
plt.plot(np.linspace(np.min(x4), np.max(x4), 20), -np.ones((20,))*s4+m4, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m4, '2$\sigma$ = $\pm$%.1fF'%s4])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.savefig('Fig_oralDevice #4.png', bbox_inches='tight',pad_inches = 0.1)

plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(ax4, y_device4, '.')
plt.plot(axs, np.polyval(ap4,axs), '--')
plt.plot(axs, axs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #4 vs Body','slope=%.2f, r$^2$=%.2f'%(ap4[0], arsq4), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
d4=y_device4 - ax4
m4=np.nanmean(d4)
s4=2*np.nanstd(d4)
plt.plot(ax4, d4, 'b.')
plt.plot(np.linspace(np.min(ax4), np.max(ax4), 20), np.ones((20,))*m4, 'k-.')
plt.plot(np.linspace(np.min(ax4), np.max(ax4), 20), np.ones((20,))*s4+m4, 'k--')
plt.plot(np.linspace(np.min(ax4), np.max(ax4), 20), -np.ones((20,))*s4+m4, 'k--')
plt.legend(['Detected - Elevated', 'CB = %.2fF'%m4, '2$\sigma$ = $\pm$%.1fF'%s4])
plt.ylabel('Detected - Elevated (F)', fontsize=16)
plt.xlabel('Elevated Temperatures (F)', fontsize=14)

plt.savefig('Fig_surfDevice #4.png', bbox_inches='tight',pad_inches = 0.1)
all_fits['device4'] = [p4, rsq4, ap4, arsq4]

plt.close()
plt.figure()
#plt.subplot(1,2,1)
plt.subplot(1,2,1)
plt.plot(x5, y_device5, '.')
xs=np.linspace(np.nanmin(x), np.nanmax(x), 100)
plt.plot(xs, np.polyval(p5,xs), '--')
#plt.plot(x5, y_device5_bt, 'r.')
#plt.plot(xs, np.polyval(p5_bt,xs), 'g--')
plt.plot(xs, xs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #5 Surface vs Body','slope=%.2f, r$^2$=%.2f'%(p5[0], rsq5), 'Body vs Body'], loc=2)
#plt.legend(['Device #5 Surf vs Alt','slope=%.2f, r$^2$=%.2f'%(p5[0], rsq5), 'Device #5 Body','slope=%.2f, r$^2$=%.2f'%(p5_bt[0], rsq5_bt), 'Body vs Body'], loc=2)
plt.subplot(1,2,2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
#d5=y_device5 - un_physio_correction(x5, 75.2)
d5=y_device5 - x5
m5=np.nanmean(d5)
s5=2*np.nanstd(d5)
plt.plot(x5, d5, 'b.')
plt.plot(np.linspace(np.min(x5), np.max(x5), 20), np.ones((20,))*s5+m5, 'k--')
plt.plot(np.linspace(np.min(x5), np.max(x5), 20), np.ones((20,))*m5, 'k-.')
plt.plot(np.linspace(np.min(x5), np.max(x5), 20), -np.ones((20,))*s5+m5, 'k--')
#plt.legend(['Detected - Elevated (Surf)', '2$\sigma$ = $\pm$%.1fF'%s5])
plt.legend(['Detected (Surf) - Elevated', 'CB = %.2fF'%m5, '2$\sigma$ = $\pm$%.1fF'%s5])
plt.ylabel('Detected - Elevated Surf (F)', fontsize=16)
plt.xlabel('Surf Temperatures (F)', fontsize=14)
plt.savefig('Fig_oralDevice #5.png', bbox_inches='tight',pad_inches = 0.1)


plt.close()
plt.figure()
plt.subplot(1,2,1)
plt.plot(ax5, y_device5, '.')
# axs is the elevated temps, so it should indeed be the same here
plt.plot(axs, np.polyval(ap5,axs), '--')
#plt.plot(axs, un_physio_correction(axs, 71.2)+1.8, 'k-.')
#plt.plot(ax5, y_device5_bt, 'r.')
#plt.plot(axs, np.polyval(ap5_bt,axs), 'g-.')
plt.plot(axs, axs, 'k-.')
plt.xlabel('Elevated Temperatures (F)', fontsize=14)
plt.ylabel('Detected Temperatures (F)', fontsize=16)
plt.legend(['Device #5 Surface vs Body','slope=%.2f, r$^2$=%.2f'%(p5[0], rsq5), 'Body vs Body'], loc=2)
#plt.legend(['Device #5 Surf vs Alt','slope=%.2f, r$^2$=%.2f'%(ap5[0], arsq5), 'Device #5 Body', 'slope=%.2f, r$^2$=%.2f'%(ap5_bt[0], arsq5_bt), 'Alt Body vs Alt Body'], loc=2)
#plt.legend(['Device #5 vs Alt Surf','slope=%.2f, r$^2$=%.2f'%(ap5[0], arsq5), 'Surf+1.8F vs Body', 'Device #5 after Eqn 2'], loc=2)
axa = plt.subplot(1,2,2)
axa.yaxis.set_label_position("right")
axa.yaxis.tick_right()
#d5=y_device5_bt - ax5
d5=y_device5 - ax5
m5=np.nanmean(d5)
s5=2*np.nanstd(d5)
plt.plot(ax5, d5, 'b.')
plt.plot(np.linspace(np.min(ax5), np.max(ax5), 20), np.ones((20,))*s5+m5, 'k--')
plt.plot(np.linspace(np.min(ax5), np.max(ax5), 20), np.ones((20,))*m5, 'k-.')
plt.plot(np.linspace(np.min(ax5), np.max(ax5), 20), -np.ones((20,))*s5+m5, 'k--')
#plt.legend(['Detected - Elevated (Surf)', '2$\sigma$ = $\pm$%.1fF'%s5])
plt.legend(['Detected (Surf) - Elevated', 'CB = %.2fF'%m5, '2$\sigma$ = $\pm$%.1fF'%s5])
plt.ylabel('Detected - Elevated Surf (F)', fontsize=16)
plt.xlabel('Surf Temperatures (F)', fontsize=14)
plt.savefig('Fig_surfDevice #5.png', bbox_inches='tight',pad_inches = 0.1)
all_fits['device5'] = [p5, rsq5, ap5, arsq5]

# get the TPR and FPR for the manufacturer-specified febrile thresholds
# the only way we have to assess device 5 post-hoc without having measured 
# control subjects between each elevated measurement is using Eqn 2
# below is the actual selection threshold, which we leave at mfg's thresholds to be fair to their specifications
det_thresholds=[thr1,thr2,thr3,thr4,thr5]
# For Device 5 however, we need to specify a reasonable febrile threshold instead of the surface temperature threshold.
febrile_thresholds=[thr1,thr2,thr3,thr4,thr5_selection]
oralbased_temps=[x1, x2, x3, x4, x5]
surfbased_temps=[ax1, ax2, ax3, ax4, ax5]
devices=[y_device1, y_device2, y_device3, y_device4, y_device5] 
# the elevated temperatures the same for all devices, but some have holes where a measurement was not acquired
print('x-axis measurements per device: ', [len(t) for t in oralbased_temps])
print('y-axis measurements per device: ', [len(t) for t in devices])
fprs_oralbased=np.zeros((5,1))
tprs_oralbased=np.zeros((5,1))
fprs_surfbased=np.zeros((5,1))
tprs_surfbased=np.zeros((5,1))
for i in range(5):
    # fpr = those actually below-threshold who were detected as above threshold divided by the number actually below threshold
    fprs_oralbased[i] = np.nansum(devices[i][oralbased_temps[i]<febrile_thresholds[i]]>det_thresholds[i])/np.nansum([oralbased_temps[i]<febrile_thresholds[i]])
    # tpr = those actually above-threshold who were detected as above threshold divided by the number actually above threshold
    tprs_oralbased[i] = np.nansum(devices[i][oralbased_temps[i]>=febrile_thresholds[i]]>det_thresholds[i])/np.nansum([oralbased_temps[i]>=febrile_thresholds[i]])
    fprs_surfbased[i] = np.nansum(devices[i][surfbased_temps[i]<febrile_thresholds[i]]>det_thresholds[i])/np.nansum([surfbased_temps[i]<febrile_thresholds[i]])
    tprs_surfbased[i] = np.nansum(devices[i][surfbased_temps[i]>=febrile_thresholds[i]]>det_thresholds[i])/np.nansum([surfbased_temps[i]>=febrile_thresholds[i]])

print('TPR-surf: ', tprs_surfbased)
print('FPR-surf: ', fprs_surfbased)
print('TPR-oral: ', tprs_oralbased)
print('FPR-oral: ', fprs_oralbased)

# plot the bias-to-normal in the quick vs monitor-mode readings of the oral thermometer
plt.close()
plt.figure()
quickoral=full_dataset[:,:,2].flatten()*1.8+32
longoral=full_dataset[:,:,1].flatten()*1.8+32
plt.plot(longoral, quickoral, '.')
lm=np.min(longoral)
lx=np.max(longoral)
plt.plot(np.linspace(lm, lx, 20), np.linspace(lm, lx,20), 'k--')
p=np.polyfit(longoral, quickoral, 1)
plt.plot(np.linspace(lm, lx, 20), np.polyval(p, np.linspace(lm, lx,20)), 'r--')
plt.legend(['Quick vs Long','1:1', 'slope=%.3f'%p[0]])
plt.xlabel('Monitor Mode Oral (F)', fontsize=14)
plt.ylabel('Quick-Read Oral (F)', fontsize=16)
plt.savefig('Oral_quick_vs_long.png', bbox_inches='tight',pad_inches = 0.1)



# and, to see the effect of increasing/decreasing thresholds (increase device 1&5 by 1.0F, decrease others)...
det_thresholds=[101.4, 98.9, 99.5, 99.5, thr5+1.0]
febrile_thresholds=[100.4, 100.4, 100.4, 100.4, 100.4]
fprs_oralbased=np.zeros((5,1))
tprs_oralbased=np.zeros((5,1))
fprs_surfbased=np.zeros((5,1))
tprs_surfbased=np.zeros((5,1))
for i in range(5):
    # fpr = those actually below-threshold who were detected as above threshold divided by the number actually below threshold
    fprs_oralbased[i] = np.nansum(devices[i][oralbased_temps[i]<febrile_thresholds[i]]>det_thresholds[i])/np.nansum([oralbased_temps[i]<febrile_thresholds[i]])
    # tpr = those actually above-threshold who were detected as above threshold divided by the number actually above threshold
    tprs_oralbased[i] = np.nansum(devices[i][oralbased_temps[i]>=febrile_thresholds[i]]>det_thresholds[i])/np.nansum([oralbased_temps[i]>=febrile_thresholds[i]])
    fprs_surfbased[i] = np.nansum(devices[i][surfbased_temps[i]<febrile_thresholds[i]]>det_thresholds[i])/np.nansum([surfbased_temps[i]<febrile_thresholds[i]])
    tprs_surfbased[i] = np.nansum(devices[i][surfbased_temps[i]>=febrile_thresholds[i]]>det_thresholds[i])/np.nansum([surfbased_temps[i]>=febrile_thresholds[i]])

print(' for detection thresholds = ', det_thresholds, ' and febrile threshold of 100.4F')
print('TPR-surf: ', tprs_surfbased)
print('FPR-surf: ', fprs_surfbased)
print('TPR-oral: ', tprs_oralbased)
print('FPR-oral: ', fprs_oralbased)
