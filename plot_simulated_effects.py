
import numpy as np
import matplotlib.pylab as plt
txtload = lambda filename: np.genfromtxt(filename, delimiter=',')
plt.ion()

# Figure showing simulated bias-to-normal
plt.clf()
data=np.random.uniform(low=35, high=40, size=100)
data=data.tolist()
data.sort()
delta=np.random.randn(100)
tempdata_nobias=data+delta/3
plt.plot(data, data-tempdata_nobias, '.')
bias=2/3.0
biased_data=bias*37.5 + (1-bias)*tempdata_nobias
plt.plot(data, data-biased_data, '.')
plt.plot(data, data-(37.5 + 0.0*tempdata_nobias), '.')
plt.plot(data, np.polyval(np.polyfit(data, data-biased_data, 1), data))
slope=round(np.polyfit(data, data-biased_data,1)[0],2)
slope_str='slope=%.2f'%(slope)
plt.legend(['no bias', '67% b2n, '+slope_str, '100% b2n, slope=1.0'])
plt.ylabel('T Reference minus T DUT')
plt.xlabel('T Reference')
plt.savefig('simbias67p.png')


# Matt's idea: test hypothesis that a fixed offset could appear like a trendline in the Bland-Altman plots

# generate surface temperatures from 32 to 37C
data=np.random.uniform(low=32, high=37, size=100)
data=data.tolist()
data.sort()
# add random noise
delta=np.random.randn(100)
surf_temps=np.array(data)+delta/3

# generate body temperatures from this surface data under the following different assumptions:
# 1. the body temperature is a fixed offset from the surface temperatures (3C)
fixed_offset=3.0
# 2. the body temperature is a proportional offset from surface/ambient ((surf-21C)*F) with F=0.1, F=0.2 and F=0.4
amb=21
F1=0.1
F2=0.2
F3=0.4
physio_correction = lambda surf, amb, F: surf + (surf-amb)*F
body_fixed=surf_temps+fixed_offset
body_prop1=physio_correction(surf_temps,amb,F1)
body_prop2=physio_correction(surf_temps,amb,F2)
body_prop3=physio_correction(surf_temps,amb,F3)
# 3. the fitted exergen and Braun curves (2nd order polynomial)
p_exergen=np.array([1.29395604e-01, -8.17865385e+00, 1.64939753e+02])
p_braun_fh=np.array([1.37751138e-01, -8.58580309e+00, 1.69421134e+02])
body_exergen=np.polyval(p_exergen, surf_temps)
body_braun_fh=np.polyval(p_braun_fh, surf_temps)

# next apply a bias-to-normal to all (except braun and exergen) and plot with/without b2n
b2n = lambda temperature, normal=37.0, alpha=0.5: alpha*normal + (1-alpha)*temperature
body_fixed_b2n = b2n(body_fixed)
body_prop1_b2n = b2n(body_prop1)
body_prop2_b2n = b2n(body_prop2)
body_prop3_b2n = b2n(body_prop3)

# test the hypothesis: fixed offset can look like bias to normal on bland-altman plots
plt.clf()
plt.plot(body_prop2, body_fixed-body_prop2, '.')
plt.plot(body_prop2, body_fixed_b2n-body_prop2, '.')
p=np.polyfit(body_prop2, body_fixed-body_prop2, 1)[0]
pb=np.polyfit(body_prop2, body_fixed_b2n-body_prop2, 1)[0]
plt.legend(['Fixed offset minus Reference, %.1fpct slope'%(p*100), 'Same but 50pct b2n, with %.1pct slope'%(pb*100)])
plt.title('Bland-Altman of Fixed Offset')
plt.xlabel('Reference (C)')
plt.ylabel('T_Offset - T_Reference (C)')
plt.savefig('effect_of_b2n_onfixed.png')

plt.clf()
plt.plot(body_prop2, body_prop2-body_prop2, '.')
plt.plot(body_prop2, body_prop2_b2n-body_prop2, '.')
pb=np.polyfit(body_prop2, body_prop2_b2n-body_prop2, 1)
plt.plot(body_prop2, body_exergen-body_prop2, '.')
plt.plot(body_prop2, body_braun_fh-body_prop2, '.')
plt.legend(['Ref-Ref (all zero)', 'Ref+b2n minus Ref, with 50pct slope', 'Exergen empirical replicated', 'Braun FH empirical replicated'])
plt.title('Bland-Altman of B2N alone and two empirical curves')
plt.xlabel('Reference (C)')
plt.ylabel('T_Offset - T_Reference (C)')
plt.savefig('effect_of_b2n_alone_and_empiricals.png')

