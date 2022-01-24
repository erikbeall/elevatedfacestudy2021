#!/usr/bin/env python3

import numpy as np
import matplotlib.pylab as plt
plt.ion()

txtload = lambda filename: np.genfromtxt(filename, delimiter=',')
# NOTE: this is NOT the correct physiologic factor for foreheads
F=0.1575
f=F/(1+F)
physio_correction = lambda surf, amb: surf + (surf-amb)*F
un_physio_correction = lambda core, amb: core - (core-amb)*f
markersize=20

# skip the header and the first three
data=txtload('reversed_curves.txt')[4:,:]

# average the surface mode data
# these were obtained from surface modes of both NCIT 1 and 2
# which dynamically agree with each other with an average difference of 0.09C (0.16F)
surf=0.5*data[:,0]+0.5* (data[:,1]-32)/1.8

# convert 2, 3 and 6 to C
data[:,2] = (data[:,2] - 32)/1.8
data[:,3] = (data[:,3] - 32)/1.8
data[:,6] = (data[:,6] - 32)/1.8

plt.clf()
# these numbers were manually collected
# in-ear scanner (much narrower range it would accept)
ncit5 = np.array([95.3, 96.1, 97.2, 98.25, 99.13, 100.1, 101.2])
setpoints=np.array([34.5, 35.0, 35.5, 36, 36.5, 37, 37.5])
surfs=setpoints*1.8+32
plt.plot(surfs, ncit5, '.', markersize=markersize)
plt.plot(surfs, surfs*1.0964 - 7.95)
plt.plot(surfs, np.ones((len(surfs),))*96.8, 'k--')
plt.plot(surfs, np.ones((len(surfs),))*100.4, 'k--')
plt.xlabel('Surface Temperature (F)', fontsize=16)
plt.ylabel('Body Temperature (F)', fontsize=16)
plt.xlabel('Surface Temperature (F)', fontsize=16)
plt.ylabel('Body Temperature (F)', fontsize=16)
plt.legend(['NCIT #5 (aural) Estimated Body Temperature', 'Surface*1.0964 - 7.95'])
plt.savefig('transfer_curve_nob2n_ncit5.png', bbox_inches='tight',pad_inches = 0.1)
plt.close()
plt.figure()
#pad_inches = 0


inds=np.isnan(data[:,2])==False
p_ncit4=np.polyfit(surf[inds], data[inds,2], 2)
# actually two b2ns - one below 36.2C surface and one above 36.0C
p_ncit4_1=np.polyfit(surf[(surf<36.2)*inds], data[(surf<36.2)*inds,2], 2)
p_ncit4_2=np.polyfit(surf[(surf>=35.8)*inds], data[(surf>=35.8)*inds,2], 2)
ncit4_b2n = lambda surf: np.polyval(p_ncit4_1, surf) if surf<36.2 else np.polyval(p_ncit4_2, surf)
ncit4_out = [ncit4_b2n(e) for e in surf]
plt.clf()
plt.plot(surf*1.8+32, data[:,2]*1.8+32, '.', markersize=markersize)
plt.plot(surf*1.8+32, physio_correction(surf, 22)*1.8+32)
plt.plot(surf*1.8+32, np.ones((len(surf),))*96.8, 'k--')
plt.plot(surf*1.8+32, [e*1.8+32 for e in ncit4_out], 'r')
plt.plot(surf*1.8+32, np.ones((len(surf),))*100.4, 'k--')
plt.xlabel('Surface Temperature (F)')
plt.ylabel('Body Temperature (F)')
plt.xlabel('Surface Temperature (F)', fontsize=16)
plt.ylabel('Body Temperature (F)', fontsize=16)
plt.legend(['NCIT #4 Mode Output','71.6F Surface-to-Core','96.8F to 100.4F "normal" range', 'Piecewise bias-to-normal'])
plt.savefig('b2n_ncit4.png', bbox_inches='tight',pad_inches = 0.1)
plt.close()
plt.figure()

plt.clf()
plt.plot(surf*1.8+32, data[:,6]*1.8+32, '.', markersize=markersize)
plt.plot(surf*1.8+32, physio_correction(surf, 22)*1.8+32)
plt.plot(surf*1.8+32, np.ones((len(surf),))*96.8, 'k--')
p_ncit3=np.polyfit(surf, data[:,6], 3)
ncit3_out = np.polyval(p_ncit3, surf)
plt.plot(surf*1.8+32, np.polyval(p_ncit3, surf)*1.8+32, 'r')
plt.plot(surf*1.8+32, np.ones((len(surf),))*100.4, 'k--')
plt.xlabel('Surface Temperature (F)')
plt.ylabel('Body Temperature (F)')
plt.xlabel('Surface Temperature (F)', fontsize=16)
plt.ylabel('Body Temperature (F)', fontsize=16)
plt.legend(['NCIT #3 Body Mode Output','71.6F Surface-to-Core','96.8F to 100.4F "normal" range', 'Single bias-to-normal fit'])
plt.savefig('b2n_ncit3.png', bbox_inches='tight',pad_inches = 0.1)
plt.close()
plt.figure()


plt.clf()
plt.plot(surf*1.8+32, data[:,3]*1.8+32, '.', markersize=markersize)
plt.plot(surf*1.8+32, physio_correction(surf, 22)*1.8+32)
plt.plot(surf*1.8+32, np.ones((len(surf),))*96.8, 'k--')
p_ncit2_2=np.polyfit(surf[(surf<36.5)*(surf>32)], data[(surf<36.5)*(surf>32),3], 1)
p_ncit2_3=np.polyfit(surf[(surf>=36.5)], data[(surf>=36.5),3], 1)
ncit2_b2n = lambda surf: np.polyval(p_ncit2_2, surf) if surf<36.5 else np.polyval(p_ncit2_3, surf)
ncit2_out = [ncit2_b2n(s) for s in surf]
plt.plot(surf*1.8+32, [e*1.8+32 for e in ncit2_out], 'r')
plt.plot(surf*1.8+32, np.ones((len(surf),))*100.4, 'k--')
plt.xlabel('Surface Temperature (F)')
plt.ylabel('Body Temperature (F)')
plt.xlabel('Surface Temperature (F)', fontsize=16)
plt.ylabel('Body Temperature (F)', fontsize=16)
plt.legend(['NCIT #2 Body Mode Output','71.6F Surface-to-Core','96.8F to 100.4F "normal" range', 'Piecewise bias-to-normal'])
plt.savefig('b2n_ncit2.png', bbox_inches='tight',pad_inches = 0.1)
plt.close()
plt.figure()

plt.clf()
plt.plot(surf*1.8+32, data[:,4]*1.8+32, '.', markersize=markersize)
plt.plot(surf*1.8+32, physio_correction(surf, 22)*1.8+32)
plt.plot(surf*1.8+32, np.ones((len(surf),))*96.8, 'k--')
p_ncit1_2=np.polyfit(surf[(surf<36.2)*(surf>32)], data[(surf<36.2)*(surf>32),4], 1)
p_ncit1_3=np.polyfit(surf[(surf>=36.2)], data[(surf>=36.2),4], 1)
ncit1_b2n = lambda surf: np.polyval(p_ncit1_2, surf) if surf<36.2 else np.polyval(p_ncit1_3, surf)
ncit1_out = [ncit1_b2n(e) for e in surf]
plt.plot(surf*1.8+32, [e*1.8+32 for e in ncit1_out], 'r')
plt.plot(surf*1.8+32, np.ones((len(surf),))*100.4, 'k--')
plt.xlabel('Surface Temperature (F)')
plt.ylabel('Body Temperature (F)')
plt.xlabel('Surface Temperature (F)', fontsize=16)
plt.ylabel('Body Temperature (F)', fontsize=16)
plt.legend(['NCIT #1 Body Mode Output','71.6F Surface-to-Core','96.8F to 100.4F "normal" range', 'Piecewise bias-to-normal'])
plt.savefig('b2n_ncit1.png', bbox_inches='tight',pad_inches = 0.1)

# get normalization ranges - these are all in Celsius
ncit1_out = np.array(ncit1_out)
ncit2_out = np.array(ncit2_out)
ncit3_out = np.array(ncit3_out)
ncit4_out = np.array(ncit4_out)
thresh=37.5
f_ind_q=np.where(ncit1_out>=thresh)[0][0]
# recordings below scale
#h_ind_q=np.where(ncit1_out<=35)[0][-1]
f_ind_ext=np.where(ncit2_out>=thresh)[0][0]

#h_ind_ext=np.where(ncit2_out<=35)[0][-1]
thresh=(100.1 - 32)/1.8
f_ind_exg=np.where(ncit3_out>=38)[0][0]
#h_ind_exg=np.where(ncit3_out<=35)[0][-1]

thresh=37.5
f_ind_bf=np.where(ncit4_out>=thresh)[0][0]
h_ind_bf=np.where(ncit4_out<=35)[0][-1]

body_temps = physio_correction(surf, 22)*1.8+32
print('Normalization range for NCIT 1: %.2f- %.2fF'%(body_temps[0], body_temps[f_ind_q-1]))
print('Normalization range for NCIT 2: %.2f - %.1fF'%(body_temps[0],body_temps[f_ind_exg-1]))
print('Normalization range for NCIT 3: %.2f - %.1fF'%(body_temps[0], body_temps[f_ind_ext-1]))
print('Normalization range for NCIT 4: %.1f - %.1fF'%(body_temps[h_ind_bf+1], body_temps[f_ind_bf-1]))

