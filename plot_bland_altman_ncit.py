#!/usr/bin/env python3

# uses the march 2023 acquisitions

import numpy as np
import matplotlib.pylab as plt
plt.ion()

# load data
txtload = lambda filename: np.genfromtxt(filename, delimiter=',')
data=txtload('ncit_measurements_multiple_ambients.csv')[1:,:]

# convert 10, 11 to C
data[:,10] = (data[:,10] - 32)/1.8
data[:,11] = (data[:,11] - 32)/1.8

# NOTE: this may not be the correct physiologic factor for each NCIT
# other issues: numerous other confounds have been raised as potentially throwing off the factor (and measurement), 
# but most are purely theoretical with little data to back up an effect for NCITs. 
# the biggest confounds will be
# 1) the surface area seen by the NCIT, 2) luck in placement of the detection window spot-on-target, and 3) forehead skin vasomotion
# Only the exergen avoids #1 by having a fixed spot-on-target, while only the Exergen maxima methods reduce #2 and #3 to some extent 
# but there's still luck involved.
# in IRT this factor can also be affected by SSE, but that's a separate technology, 
# plus they ought to be correcting for SSE effects (if they want their IRT to give diagnostically relevant decisions)
# Regardless, we can assume there is a physiologic factor per skin site, and there may be other device factors that will appear to behave as if conflated with this
# and may or may not be disentangleable from the actual physiologic factor
# IRT-derived F-values: exergen is F=0.20315, NCIT other is 0.210515
F=0.203
f=F/(1+F)
physio_correction = lambda surf, amb: surf + (surf-amb)*F
physio_correction = lambda surf, amb: surf*(1+F) - amb*F
reverse_physio_correction = lambda core, amb: core - (core-amb)*f

# generate the assumed T_Reference - body from the surface temperature presented
T_Reference_ambdep = physio_correction(data[:,0], data[:,13])
T_Reference = physio_correction(data[:,0], np.nanmean(data[:,13]))

# two different QY and two different Extech units were used, hence the two values below. All measurements were >= 3x and in most cases no change was seen (then used 4x and recorded the average), except for the Braun Notouch because it always increases by unexpectedly large amounts from scan to scan.
# SetPoint,QSurf,QSurf2,ExtechSurf,Extec2,QBody,QBody2,ExtechBody,Extec2,Exergen,Braun NoTouch,Braun In-ear,Alt,Ambient,Touch thermometer
#   0       1     2        3          4    5     6        7        8      9       10             11         12   13        14
inds1=np.where(np.abs(data[:,13]-26)<1)[0]
inds2=np.where(np.abs(data[:,13]-22)<1)[0]
inds3=np.where(np.abs(data[:,13]-19.5)<1)[0]
# exclude the <32C setpoint data
inds2=inds2[4:]

ambs1=data[inds1,13]
ambs2=data[inds2,13]
ambs3=data[inds3,13]
amb1=np.mean(data[inds1,13])
amb2=np.mean(data[inds2,13])
amb3=np.mean(data[inds3,13])
surf1=data[inds1,0]
surf2=data[inds2,0]
surf3=data[inds3,0]

# Calibration target was set in 0.5C increments between 32 and 39C (started at 30C for first set of data), 98% emissivity
# Observations: 
#   - the QY exhibited a closer adherence to setpoint (smaller delta between measured and setpoint) but a negative trendline of 100 and 200mK
#   - the Extechs exhibited a stronger ambient dependence of mean offset measured surface temp vs setpoint, but minimal trend (61mK total over 32-39C range)

trend_qy1=np.polyfit(data[:,0], data[:,1]-data[:,0], 1)[0]
trend_qy2=np.polyfit(data[:,0], data[:,2]-data[:,0], 1)[0]
print('Trend in QY2 (worst offender) produces drift from 32C to 39C of ', trend_qy2*32-trend_qy2*39)
# 0.21409193408499672
print('Trend in QY1 (second worst) produces drift from 32C to 39C of ', trend_qy1*32-trend_qy1*39)
# 0.12827059843885524

#np.polyfit(data[inds1,0], data[inds1,1]-data[inds1,0], 1)[0]
#-0.02500000000000034
#np.polyfit(data[inds2,0], data[inds2,1]-data[inds2,0], 1)[0]
#-0.009824561403508892
#np.polyfit(data[inds3,0], data[inds3,1]-data[inds3,0], 1)[0]
#-0.03642857142857107
#>>> np.polyfit(data[inds1,0], data[inds1,2]-data[inds1,0], 1)[0]
#-0.04571428571428521
#>>> np.polyfit(data[inds2,0], data[inds2,2]-data[inds2,0], 1)[0]
#-0.029122807017544244
#>>> np.polyfit(data[inds3,0], data[inds3,2]-data[inds3,0], 1)[0]
#-0.03071428571428615

# This presents us two choices, one could rely on surface temperature reported by the device, in which case no grid (which is fine)
# or rely on the setpoint, which is certainly trustworthy to some extent, 
# but note there could be fluctuations in reflected temperature and drafts, despite all precautions
# The first option is only available for the QY and Extech devices since they were the only ones to produce a surface temperature.
# What we need to avoid is analyzing it both ways and then picking blindly the one we feel fits our conclusions better. 
# For consistency, will use setpoint hereafter in all bland-altman plots. 
# However, to first test whether there is an ambient dependence in the offset (e.g. the manufacturer applying an offset that depends on ambient), 
# we first look at ambient dependence using the surface temperature produced by the device

# ambient_dependence -> systematic change in offset bw body and surface vs ambient temperature
# stack the two NCITs with two devices each
qy_surf1=np.hstack([data[inds1,1], data[inds1,2]])
qy_surf2=np.hstack([data[inds2,1], data[inds2,2]])
qy_surf3=np.hstack([data[inds3,1], data[inds3,2]])
qy_body1=np.hstack([data[inds1,5], data[inds1,6]])
qy_body2=np.hstack([data[inds2,5], data[inds2,6]])
qy_body3=np.hstack([data[inds3,5], data[inds3,6]])

ex_surf1=np.hstack([data[inds1,3], data[inds1,4]])
ex_surf2=np.hstack([data[inds2,3], data[inds2,4]])
ex_surf3=np.hstack([data[inds3,3], data[inds3,4]])
ex_body1=np.hstack([data[inds1,7], data[inds1,8]])
ex_body2=np.hstack([data[inds2,7], data[inds2,8]])
ex_body3=np.hstack([data[inds3,7], data[inds3,8]])

# no apparent ambient dependence for either QY or Extech with either analysis
# from here forward, we use the QY and Extech devices that had the closest fidelity to setpoint (indices 2 and 6 for QY, 3 and 7 for Ex)
qy_body1=data[inds1, 6]
qy_body2=data[inds2, 6]
qy_body3=data[inds3, 6]
ex_body1=data[inds1, 7]
ex_body2=data[inds2, 7]
ex_body3=data[inds3, 7]

# other devices did not always produce a reading at ends of scale so trim those readings
nullinds=data<=0
data[nullinds]=np.nan
exergen_body1=data[inds1, 9]
exergen_body2=data[inds2, 9]
exergen_body3=data[inds3, 9]
braun_forehead1=data[inds1, 10]
braun_forehead2=data[inds2, 10]
braun_forehead3=data[inds3, 10]
braun_inear1=data[inds1, 11]
braun_inear2=data[inds2, 11]
braun_inear3=data[inds3, 11]

# the b2n is very simple in all four cases: an alpha-blend explains the observations (within a limited temperature range)
# uncorrect a dataset to obtain its actual penultimate surface temperature
#b2n = lambda temp, alpha, normal=37: normal*alpha + (1-alpha)*temp
#unbias = lambda body, alpha, normal=37: (body - normal*abs(alpha))/(1-abs(alpha))
#T_surf = (T_body - T_normal*alpha)/(1-alpha) - T_offset
#T_offset = (alpha*T_normal - (1-alpha)*T_surf)/(1-alpha)

# evaluate for meaningful ambient dependence (monotonic change across ambients, change across ambients larger than 0.3C)
p1=np.polyfit(surf1[braun_forehead1>0], braun_forehead1[braun_forehead1>0], 2)
p2=np.polyfit(surf2[braun_forehead2>0], braun_forehead2[braun_forehead2>0], 2)
p3=np.polyfit(surf3[braun_forehead3>0], braun_forehead3[braun_forehead3>0], 2)
p1_b=np.polyfit(surf1[braun_forehead1>0], braun_forehead1[braun_forehead1>0], 2)
p2_b=np.polyfit(surf2[braun_forehead2>0], braun_forehead2[braun_forehead2>0], 2)
p3_b=np.polyfit(surf3[braun_forehead3>0], braun_forehead3[braun_forehead3>0], 2)
vals=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]
ambient_dep_braun_fh=np.polyfit([amb1, amb2, amb3], vals, 1)
ambient_ptp_braun_fh=np.ptp(vals)
ambient_mon_braun_fh=(vals[1]>vals[0])*(vals[2]>vals[1]) or (vals[1]<vals[0])*(vals[2]<vals[1])
vals_braun_fh=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]

p1=np.polyfit(surf1[qy_body1>0], qy_body1[qy_body1>0], 2)
p2=np.polyfit(surf2[qy_body2>0], qy_body2[qy_body2>0], 2)
p3=np.polyfit(surf3[qy_body3>0], qy_body3[qy_body3>0], 2)
vals=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]
ambient_dep_qingyuan=np.polyfit([amb1, amb2, amb3], vals, 1)
ambient_ptp_qingyuan=np.ptp(vals)
ambient_mon_qingyuan=(vals[1]>vals[0])*(vals[2]>vals[1]) or (vals[1]<vals[0])*(vals[2]<vals[1])
vals_qingyuan=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]

p1=np.polyfit(surf1[ex_body1>0], ex_body1[ex_body1>0], 2)
p2=np.polyfit(surf2[ex_body2>0], ex_body2[ex_body2>0], 2)
p3=np.polyfit(surf3[ex_body3>0], ex_body3[ex_body3>0], 2)
vals=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]
ambient_dep_extech=np.polyfit([amb1, amb2, amb3], vals, 1)
ambient_ptp_extech=np.ptp(vals)
ambient_mon_extech=(vals[1]>vals[0])*(vals[2]>vals[1]) or (vals[1]<vals[0])*(vals[2]<vals[1])
vals_extech=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]

p1=np.polyfit(surf1[exergen_body1>0], exergen_body1[exergen_body1>0], 2)
p2=np.polyfit(surf2[exergen_body2>0], exergen_body2[exergen_body2>0], 2)
p3=np.polyfit(surf3[exergen_body3>0], exergen_body3[exergen_body3>0], 2)
p1_e=np.polyfit(surf1[exergen_body1>0], exergen_body1[exergen_body1>0], 2)
p2_e=np.polyfit(surf2[exergen_body2>0], exergen_body2[exergen_body2>0], 2)
p3_e=np.polyfit(surf3[exergen_body3>0], exergen_body3[exergen_body3>0], 2)
vals=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]
ambient_dep_exergen=np.polyfit([amb1, amb2, amb3], vals, 1)
ambient_ptp_exergen=np.ptp(vals)
ambient_mon_exergen=(vals[1]>vals[0])*(vals[2]>vals[1]) or (vals[1]<vals[0])*(vals[2]<vals[1])
vals_exergen=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]

p1=np.polyfit(surf1[braun_inear1>0], braun_inear1[braun_inear1>0], 2)
p2=np.polyfit(surf2[braun_inear2>0], braun_inear2[braun_inear2>0], 2)
p3=np.polyfit(surf3[braun_inear3>0], braun_inear3[braun_inear3>0], 2)
vals=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]
ambient_dep_braun_inear=np.polyfit([amb1, amb2, amb3], vals, 1)
ambient_ptp_braun_inear=np.ptp(vals)
ambient_mon_braun_inear=(vals[1]>vals[0])*(vals[2]>vals[1]) or (vals[1]<vals[0])*(vals[2]<vals[1])
vals_braun_inear=[np.polyval(p1, 35), np.polyval(p2, 35), np.polyval(p3, 35)]

# variance explained by ambient temperature fit
# difference in variance between raw data de-trended and raw data minus ambient dependence and same detrend
plt.figure()
# these were manually ordered descending by first datapoint, to enhance readability
plt.plot([amb1, amb2, amb3], vals_braun_fh, 'b.', markersize=12)
plt.plot([amb1, amb2, amb3], vals_exergen, 'r.', markersize=12)
plt.plot([amb1, amb2, amb3], vals_extech, 'g.', markersize=12)
plt.plot([amb1, amb2, amb3], vals_qingyuan, 'k.', markersize=12)
plt.plot([amb1, amb2, amb3], vals_braun_inear, 'm.', markersize=12)

plt.plot([amb1, amb2, amb3], vals_braun_fh, 'b', markersize=12)
plt.plot([amb1, amb2, amb3], vals_exergen, 'r', markersize=12)
plt.plot([amb1, amb2, amb3], vals_extech, 'g', markersize=12)
plt.plot([amb1, amb2, amb3], vals_qingyuan, 'k', markersize=12)
plt.plot([amb1, amb2, amb3], vals_braun_inear, 'm', markersize=12)

plt.legend(['NCIT #4', 'NCIT #3','NCIT #2','NCIT #1', 'NCIT #5'])
plt.title('Mean Body Temperatures versus Ambient (descending order)')
plt.ylabel('Mean Body (Matched Setpoints) (C)')
plt.xlabel('Ambient (C)')
plt.savefig('ambient_curves.png')
plt.close()
print('Table 2: ptp, slope, monotonicity')
print(round(ambient_ptp_qingyuan, 2), round(ambient_dep_qingyuan[0], 3), ambient_mon_qingyuan)
print(round(ambient_ptp_extech, 2), round(ambient_dep_extech[0], 3), ambient_mon_extech)
print(round(ambient_ptp_exergen, 2), round(ambient_dep_exergen[0], 3), ambient_mon_exergen)
print(round(ambient_ptp_braun_fh, 2), round(ambient_dep_braun_fh[0], 3), ambient_mon_braun_fh)
print(round(ambient_ptp_braun_inear, 2), round(ambient_dep_braun_inear[0], 3), ambient_mon_braun_inear)


# Supplement S.2 ambient dependence figures
plt.figure()
plt.plot(surf1[:-1], qy_body1[:-1], 'r.', markersize=12)
plt.plot(surf2[:-1], qy_body2[:-1], 'g.', markersize=12)
plt.plot(surf3[:-1], qy_body3[:-1], 'b.', markersize=12)
plt.plot(surf1[:-1], qy_body1[:-1], 'r')
plt.plot(surf2[:-1], qy_body2[:-1], 'g')
plt.plot(surf3[:-1], qy_body3[:-1], 'b')
plt.legend(['Ambient 19C', 'Ambient 22C','Ambient 26.5C'])
plt.xlabel('Setpoints (C)')
plt.ylabel('Body Output (C)')
plt.title('NCIT #1 Body vs Surface at 3 ambients')
plt.savefig('qy_curves.png')
plt.close()
plt.figure()
plt.plot(surf1[:-1], ex_body1[:-1], 'r.', markersize=12)
plt.plot(surf2[:-1], ex_body2[:-1], 'g.', markersize=12)
plt.plot(surf3[:-1], ex_body3[:-1], 'b.', markersize=12)
plt.plot(surf1[:-1], ex_body1[:-1], 'r')
plt.plot(surf2[:-1], ex_body2[:-1], 'g')
plt.plot(surf3[:-1], ex_body3[:-1], 'b')
plt.legend(['Ambient 19C', 'Ambient 22C','Ambient 26.5C'])
plt.xlabel('Setpoints (C)')
plt.ylabel('Body Output (C)')
plt.title('NCIT #2 Body vs Surface at 3 ambients')
plt.savefig('ex_curves.png')
plt.close()
# for those two devices with significant dependence, show the fits for each ambient
plt.figure()
plt.plot(surf1[:-1], exergen_body1[:-1], 'r.', markersize=12)
plt.plot(surf2[:-1], exergen_body2[:-1], 'g.', markersize=12)
plt.plot(surf3[:-1], exergen_body3[:-1], 'b.', markersize=12)
plt.plot(surf1[:-1], np.polyval(p1_e, surf1[:-1]), 'r')
plt.plot(surf2[:-1], np.polyval(p2_e, surf2[:-1]), 'g')
plt.plot(surf3[:-1], np.polyval(p3_e, surf3[:-1]), 'b')
plt.legend(['Ambient 19C', 'Ambient 22C','Ambient 26.5C'])
plt.xlabel('Setpoints (C)')
plt.ylabel('Body Output (C)')
plt.title('NCIT #3 Body vs Surface at 3 ambients')
plt.savefig('exergen_curves.png')
plt.close()
plt.figure()
plt.plot(surf1[:-1], braun_forehead1[:-1], 'r.', markersize=12)
plt.plot(surf2[:-1], braun_forehead2[:-1], 'g.', markersize=12)
plt.plot(surf3[:-1], braun_forehead3[:-1], 'b.', markersize=12)
plt.plot(surf1[:-1], np.polyval(p1_b, surf1[:-1]), 'r')
plt.plot(surf2[:-1], np.polyval(p2_b, surf2[:-1]), 'g')
plt.plot(surf3[:-1], np.polyval(p3_b, surf3[:-1]), 'b')
plt.legend(['Ambient 19C', 'Ambient 22C','Ambient 26.5C'])
plt.xlabel('Setpoints (C)')
plt.ylabel('Body Output (C)')
plt.title('NCIT #4 Body vs Surface at 3 ambients')
plt.savefig('braun_fh_curves.png')
plt.close()
plt.figure()
plt.plot(surf1[:-1], braun_inear1[:-1], 'r.', markersize=12)
plt.plot(surf2[:-1], braun_inear2[:-1], 'g.', markersize=12)
plt.plot(surf3[:-1], braun_inear3[:-1], 'b.', markersize=12)
plt.plot(surf1[:-1], braun_inear1[:-1], 'r')
plt.plot(surf2[:-1], braun_inear2[:-1], 'g')
plt.plot(surf3[:-1], braun_inear3[:-1], 'b')
plt.legend(['Ambient 19C', 'Ambient 22C','Ambient 26.5C'])
plt.xlabel('Setpoints (C)')
plt.ylabel('Body Output (C)')
plt.title('NCIT #5 Body vs Surface at 3 ambients')
plt.savefig('braun_inear_curves.png')
plt.close()

# pseudo bland-altman curves
markersize=20
surf=data[:,0]

# ncit 1 = column 6 = Qingyuan #2
temps=data[:,6]
T_Body_NCIT1 = data[:,6]

# ncit 2 = column 7 = Extech #1
temps=data[:,7]
T_Body_NCIT2 = data[:,7]

# ncit 3 = column 9 = Exergen
temps=data[:,9]
T_Body_NCIT3 = data[:,9]

# ncit 4 = column 10 = Braun forehead
temps=data[:,10]
T_Body_NCIT4 = data[:,10]

# ncit 5 = in-ear Braun scanner
temps=data[:,11]
T_Body_NCIT5 = data[:,11]

# smaller correction factor for in-ear
F_inear=0.12377
physio_correction_inear = lambda surf, amb: surf + (surf-amb)*F_inear
T_Reference_NCIT5 = physio_correction_inear(surf, 24) 
T_Reference_NCIT5_ambdep = physio_correction_inear(surf, data[:,13])
# F-value for Braun Thermoscan in-ear: F=0.12377

# Bland-Altman and fit our assumed T_Reference
# plotting T_DUT - T_Reference (reverse of Allegaert) vs T_Reference and include fit(s)
# fit from 35 to 39C T_Reference
inds_fit=np.where((T_Reference<=39) * (T_Reference>=35.0))[0]

plt.clf()
plt.plot(T_Reference, T_Body_NCIT1 - T_Reference, '.', markersize=markersize)
p_ncit1 = np.polyfit(T_Reference[inds_fit], T_Body_NCIT1[inds_fit] - T_Reference[inds_fit],1)
plt.plot(T_Reference, np.polyval(p_ncit1, T_Reference), 'k--')
plt.title('Pseudo Bland-Altman NCIT 1')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit1[0]])
plt.ylabel('T DUT minus T Reference')
plt.xlabel('T Reference')
plt.savefig('NCIT_paper_ncit1.png')
plt.clf()

plt.clf()
plt.plot(T_Reference, T_Body_NCIT2 - T_Reference, '.', markersize=markersize)
p_ncit2 = np.polyfit(T_Reference[inds_fit], T_Body_NCIT2[inds_fit] - T_Reference[inds_fit],1)
plt.plot(T_Reference, np.polyval(p_ncit2, T_Reference), 'k--')
plt.title('Pseudo Bland-Altman NCIT 2')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit2[0]])
plt.ylabel('T DUT minus T Reference')
plt.xlabel('T Reference')
plt.savefig('NCIT_paper_ncit2.png')

plt.clf()
inds_fit=np.where((T_Reference<=39) * (T_Reference>=35.0) * (np.isnan(T_Body_NCIT3)==False))[0]
inds_plt=np.where((T_Reference>=32.5) * (np.isnan(T_Body_NCIT3)==False))[0]
plt.plot(T_Reference[inds_plt], T_Body_NCIT3[inds_plt] - T_Reference[inds_plt], '.', markersize=markersize)
p_ncit3 = np.polyfit(T_Reference[inds_fit], T_Body_NCIT3[inds_fit] - T_Reference[inds_fit],1)
plt.plot(T_Reference[inds_plt], np.polyval(p_ncit3, T_Reference)[inds_plt], 'k--')
plt.title('Pseudo Bland-Altman NCIT 3')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit3[0]])
plt.ylabel('T DUT minus T Reference')
plt.xlabel('T Reference')
plt.savefig('NCIT_paper_ncit3.png')
plt.clf()

plt.clf()
inds_fit=np.where((T_Reference<=38) * (T_Reference>=35.0) * (np.isnan(T_Body_NCIT3)==False)  * (np.abs(data[:,13]-22)<1))[0]
inds_plt=np.where((T_Reference>=35) * (T_Reference<=39) * (np.isnan(T_Body_NCIT3)==False)  * (np.abs(data[:,13]-22)<1))[0]
plt.plot(T_Reference[inds_plt], (T_Reference- T_Body_NCIT3)[inds_plt], '.')
p_ncit3_alt = np.polyfit(T_Reference[inds_fit], (T_Reference- T_Body_NCIT3)[inds_fit],1)
plt.plot(T_Reference[inds_plt], np.polyval(p_ncit3_alt, T_Reference)[inds_plt])
plt.title('Pseudo Bland-Altman NCIT 3')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit3_alt[0]])
plt.ylabel('T Reference minus T DUT')
plt.xlabel('T Reference')
plt.savefig('NCIT_paper_ncit3_for_Allegaert_inverted.png')

plt.clf()
inds_fit=np.where((T_Reference<=38) * (T_Reference>=33.0) * (np.isnan(T_Body_NCIT4)==False))[0]
plt.plot(T_Reference, T_Body_NCIT4 - T_Reference, '.', markersize=markersize)
p_ncit4 = np.polyfit(T_Reference[inds_fit], T_Body_NCIT4[inds_fit] - T_Reference[inds_fit],1)
plt.plot(T_Reference, np.polyval(p_ncit4, T_Reference), 'k--')
plt.title('Pseudo Bland-Altman NCIT 4')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit4[0]])
plt.ylabel('T DUT minus T Reference')
plt.xlabel('T Reference')
plt.savefig('NCIT_paper_ncit4.png')
# second form with ambient-dependent reference, instead of using a single fixed ambient to generate the reference (needed because the Braun actually does apply an ambient correction)
plt.clf()
plt.plot(T_Reference_ambdep, T_Body_NCIT4 - T_Reference_ambdep, '.', markersize=markersize)
p_ncit4 = np.polyfit(T_Reference_ambdep[inds_fit], T_Body_NCIT4[inds_fit] - T_Reference_ambdep[inds_fit],1)
plt.plot(T_Reference_ambdep, np.polyval(p_ncit4, T_Reference_ambdep), 'k--')
plt.title('Pseudo Bland-Altman NCIT 4')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit4[0]])
plt.ylabel('T DUT minus T Reference')
plt.xlabel('T Reference - Ambient-Corrected')
plt.savefig('NCIT_paper_ncit4_withambdep.png')

plt.clf()
# fit using the reference intended for in-ear
T_Reference=T_Reference_NCIT5 
inds_fit=np.where((T_Reference<=39) * (T_Reference>=np.nanmin(T_Body_NCIT5)) * (np.isnan(T_Body_NCIT5)==False))[0]
inds_plt=np.where((T_Reference>=np.nanmin(T_Body_NCIT5)) * (np.isnan(T_Body_NCIT5)==False))[0]
plt.plot(T_Reference[inds_plt], (T_Body_NCIT5 - T_Reference)[inds_plt], '.', markersize=markersize)
p_ncit5 = np.polyfit(T_Reference[inds_fit], T_Body_NCIT5[inds_fit] - T_Reference[inds_fit],1)
plt.plot(T_Reference[inds_plt], np.polyval(p_ncit5, T_Reference)[inds_plt], 'k--')
plt.title('Pseudo Bland-Altman NCIT 5')
plt.legend(['Delta T vs T', 'slope=%.2f'%p_ncit5[0]])
plt.ylabel('T DUT minus T Reference')
plt.xlabel('T Reference')
plt.savefig('NCIT_paper_ncit5.png')

print(np.polyval(p_ncit1, 37))
print(np.polyval(p_ncit2, 37))
print(np.polyval(p_ncit3, 37))
print(np.polyval(p_ncit4, 37))
print(np.polyval(p_ncit5, 37))

alphas=[p_ncit1[0], p_ncit2[0], p_ncit3[0], p_ncit4[0], p_ncit5[0]]
#np.save('alphas.npy', alphas)

