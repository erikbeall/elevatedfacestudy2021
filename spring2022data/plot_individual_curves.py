
import numpy as np
import matplotlib.pylab as plt
plt.ion()
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()

scan_data=dsload('stolaf_april2022_data.npy')
offsets={}
for unit in [49, 50, 51, 52, 54]:
    offsets[str(unit)]=[]

for uuid in scan_data.keys():
    for tent in ['A','B','C']:
        ds=scan_data[uuid][tent]
        if ds is None:
            continue
        #np.nanmedian(ds['face_temps'][-20:]) - np.nanmedian(ds['POSTEQ']['face_temps']), np.nanmedian(ds['face_temps'][-20:]) - np.nanmedian(ds['POSTEQ']['face_temps']), np.nanmedian(ds['ambients']), np.nanmedian(ds['distances']), len(ds['face_temps'])])
        plt.clf()
        try:
         plt.plot(ds['times'][1:], ds['face_temps']);
        except:
            print('ds fail: ', uuid, tent)
        try:
         plt.plot(ds['PREEQ']['times'][1:], ds['PREEQ']['face_temps']); plt.plot(ds['POSTEQ']['times'][1:], ds['POSTEQ']['face_temps']);
        except:
            print('pre/post fail: ', uuid, tent)
        try:
         plt.plot(ds['POSTEQ']['flir_times'], ds['POSTEQ']['flir'], 'k'); plt.plot(ds['PREEQ']['flir_times'], ds['PREEQ']['flir'], 'k');
        except:
            print('pre/post flir fail: ', uuid, tent)
        plt.title('id %s, Tent %s, N %d'%(uuid, tent, len(ds['face_temps'])))
        plt.savefig('data_%s_%s.png'%(uuid,tent))


for uuid in scan_data.keys():
    tent='F'
    ds=scan_data[uuid][tent]
    if ds is None:
        continue
    plt.clf()
    plt.plot(ds['times'][1:], ds['face_temps']); 
    plt.plot(ds['times'][1:], ds['face_temps']);
    plt.plot(ds['PREEQ']['times'][1:], ds['PREEQ']['face_temps']); plt.plot(ds['POSTEQ']['times'][1:], ds['POSTEQ']['face_temps']);
    plt.plot(ds['POSTEQ']['flir_times'], ds['POSTEQ']['flir'], 'k'); plt.plot(ds['PREEQ']['flir_times'], ds['PREEQ']['flir'], 'k');
    plt.title('id %s, Tent %s, N %d'%(uuid, tent, len(ds['face_temps'])))
    plt.savefig('data_%s_%s.png'%(uuid,tent))


