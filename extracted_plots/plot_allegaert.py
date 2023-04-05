
import numpy as np
import matplotlib.pylab as plt
txtload = lambda filename: np.genfromtxt(filename, delimiter=',')
plt.ion()

plt.figure()

plt.clf()
data=txtload('allegaert_fig1.csv')
plt.plot(data[:,0], data[:,1],'.')
plt.plot(data[:,0], np.polyval(np.polyfit(data[:,0], data[:,1],1), data[:,0]))
plt.title('Allegaert Fig 1')
slope=np.polyfit(data[:,0], data[:,1],1)[0]
plt.legend(['Delta T vs T', 'slope=%.2f'%slope])
plt.ylabel('T Reference minus T DUT')
plt.xlabel('T Reference')
plt.savefig('allegaert_fig1.png')

plt.clf()
data=txtload('allegaert_fig2.csv')
plt.plot(data[:,0], data[:,1],'.')
plt.plot(data[:,0], np.polyval(np.polyfit(data[:,0], data[:,1],1), data[:,0]))
plt.title('Allegaert Fig 2')
slope=np.polyfit(data[:,0], data[:,1],1)[0]
plt.legend(['Delta T vs T', 'slope=%.2f'%slope])
plt.ylabel('T Reference minus T DUT')
plt.xlabel('T Reference')
plt.savefig('allegaert_fig2.png')

plt.clf()
data=txtload('allegaert_fig3.csv')
plt.plot(data[:,0], data[:,1],'.')
plt.plot(data[:,0], np.polyval(np.polyfit(data[:,0], data[:,1],1), data[:,0]))
plt.title('Allegaert Fig 3')
slope=np.polyfit(data[:,0], data[:,1],1)[0]
plt.legend(['Delta T vs T', 'slope=%.2f'%slope])
plt.ylabel('T Reference minus T DUT')
plt.xlabel('T Reference')
plt.savefig('allegaert_fig3.png')

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


