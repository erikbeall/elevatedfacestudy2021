
import numpy as np
import matplotlib.pylab as plt
plt.ion()
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

# data acquired in 2021 used a simpler control loop that led to larger swings in the effective environmental temperature in the warmer tents especially
curves_fsurfs, curves_fenvironment, curves_fdistances, curves_fambient, curves_fradiative = np.load('../stolaf_filtered_curve_data.npy')
# for example, here is the surface temperature measurements made over 600 seconds in tent A (coolest but still warmer than the classroom environment)
plt.figure()
plt.subplot(2,1,1);
plt.plot(curves_fsurfs[0,0,:])
plt.title('TentA Subject 1')
plt.subplot(2,1,2)
plt.title('TentA Env temperture')
plt.plot(curves_fenvironment[0,0,:])

plt.figure()
plt.subplot(2,1,1);
plt.plot(curves_fsurfs[0,3,:])
plt.title('TentD Subject 1')
plt.subplot(2,1,2)
plt.title('TentD Env temperture')
plt.plot(curves_fenvironment[0,3,:])
# the oscillations were nearly 2 degrees C, which can be seen by the impact on the surface temperature while it equilibrated
# it was equilibrating to a "moving target", which isn't the subject of this study 
# (although its very interesting - the time constant is likely subject-dependent and is useful for setting a minimum equilibration time requirement)
# also note the fenvironment was constructed by a weighted sum of the fradiative and fambient (the "f" denotes minor temporal filtering applied to the raw data)
# radiative was measured background (image-based) and ambient was instrument-monitored

# the 2022 data had less fluctuations but we also skipped the hottest tent and ran for 15 minutes in each
# note pre-equilibration data for subject 110 in tent C was missing (pre-equil expected to be roughly comparable across tents, plusminus measurement fluctuation)
# and subject 101 is missing tent C data
scandata=dsload('../spring2022data/stolaf_april2022_data.npy')
pre_skintemp = np.mean(scandata['111']['A']['PREEQ']['face_temps'])
curves_skintemp = np.mean(scandata['111']['A']['face_temps'])
# note the ambient temperature in PREEQ and POSTEQ is of the room temperature
pre_ambient = np.mean(scandata['111']['A']['PREEQ']['ambients'])
post_skintemp = np.mean(scandata['111']['A']['POSTEQ']['face_temps'])
# ambient temperature inside tent can fluctuate, take final value
tent_ambient= scandata['111']['A']['ambients'][-1]
# oral values are obtained from appropriate column in orals, convert to C
# quick and dirty - 99F -> C
oral = (99-32)/1.8
# correction for Fourier static heat equation assuming the predominant effect is insulative skin and insulative air film (no significant convection)
# F-value for relative insulation from earlier empirical work - goal is to derive 1) F-value and 2) estimate systematic error
F=0.1595
f=F/(F+1)
physio_correction = lambda surf, amb, F=0.1595: surf + (surf-amb)*F
un_physio_correction = lambda core, amb, f=F/(F+1): core - (core-amb)*f
print('Ambient temperature before=%.2fC, tent=%.2fC, skin temperature before=%.2fC, tent=%.2fC'%(pre_ambient, tent_ambient, pre_skintemp, post_skintemp))
print('Oral temperature: %.2fC, core estimate before=%.2fC, after=%.2fC'%(oral, physio_correction(pre_skintemp, pre_ambient), physio_correction(post_skintemp, tent_ambient)))

plt.figure()
plt.subplot(2,1,1);
plt.plot(scandata['101']['B']['face_temps'])
plt.title('Tent B Subject 101')
plt.subplot(2,1,2)
plt.title('Tent B Env temperture')
plt.plot(scandata['101']['B']['ambients'])



