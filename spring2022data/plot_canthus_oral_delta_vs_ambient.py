
# 1. reproduce plot in paper - bland-altman
# 2. reproduce oral temperature data, combine
# 3. forehead in this data

scan_data=dsload('stolaf_april2022_data.npy')

oral_data=np.genfromtxt('oral_april2022.txt', delimiter=',', usecols=[1,2,3,4,5,6,7])
oral_data=np.nanmean(oral_data[1:,[0,4,5,6]], 1)
oral_uuids=np.genfromtxt('oral_april2022.txt', delimiter=',', usecols=[0])[1:]

# loop over uuids, we have one oral core per UUID (effectively no change in body temperature by pre-analysis)
# produce distribution of T_Surface (both flir and FI) - T_Oral versus T_Ambient
# and finally data explorer to get same out of raw image data
#F = (T_core - T_surf) / (T_surf - T_amb)
# plotting this fraction is NOT going to give good results, errors blow up as T_amb gets large and closer to T_surf

surfs_fi_canthus=[]
surfs_flir=[]
ambients=[]
orals=[]
for uuid in oral_uuids:
    oral = (oral_data[np.where(oral_uuids==uuid)[0]] - 32)/1.8
    uuid=str(int(uuid))
    for tent in ['A', 'B', 'C']:
        if scan_data[str(uuid)][tent] is None:
            continue
        tent_ambs=scan_data[str(uuid)][tent]['ambients']
        tent_ambs=tent_ambs[len(tent_ambs)//2:]
        if 'PREEQ' in scan_data[str(uuid)][tent].keys():
            surfs_fi_canthus.append(np.nanmedian(scan_data[str(uuid)][tent]['PREEQ']['face_temps']))
            surfs=scan_data[str(uuid)][tent]['PREEQ']['face_temps']
            surfs_fi_canthus.append(np.nanmedian(surfs))
            flir=scan_data[str(uuid)][tent]['PREEQ']['flir']
            surfs_flir.append(np.nanmedian([f for f in flir if f>33.5]))
            ambients.append(np.nanmean(scan_data[str(uuid)][tent]['PREEQ']['ambients']))
            orals.append(oral)
        if 'POSTEQ' in scan_data[str(uuid)][tent].keys():
            surfs_fi_canthus.append(np.nanmedian(scan_data[str(uuid)][tent]['POSTEQ']['face_temps']))
            flir=scan_data[str(uuid)][tent]['POSTEQ']['flir']
            surfs_flir.append(np.nanmedian([f for f in flir if f>33.5]))
            ambients.append(np.nanmean(tent_ambs))
            #ambients.append(np.nanmean(scan_data[str(uuid)][tent]['POSTEQ']['ambients']))
            orals.append(oral)

surfs_fi_canthus=np.array(surfs_fi_canthus)
surfs_flir=np.array(surfs_flir)
orals=np.array(orals).reshape((-1,))
ambients=np.array(ambients)

# removal of datapoints above oral temps (active cooling) and heavy sweating - anything with ambient above 32C should be removed, we can't rely on background
# other bad indices: 
inds_good_conditions=np.where((sweat[:,6]<2)*(new_ambients<32.0))[0]


inds=(ambients<32.5)*(orals>36.5)
finds=(ambients<32.5)*(orals>36.5)*(surfs_flir>0)
F_values_fi = (orals - surfs_fi_canthus)/(surfs_fi_canthus-ambients)
F_values_flir = (orals - surfs_flir)/(surfs_flir-ambients)
F_fi = np.median(F_values_fi[inds])
# 0.12509074618590083
F_flir = np.median(F_values_flir[finds])
# 0.11541604986471904
# note, FLIR uses the full face maxima, which will also notate hot arteries in neck, forehead, edges of hair-protected skin
# while FI uses a face object detector trained to produce a box around the eyes region with the same type of data


physio_correction = lambda surf, amb, F=0.125: (F+1)*surf - F*amb

# generating the plot ...
plt.plot(ambients,  orals - (surfs_fi_canthus + np.polyval(p_offset_fi, ambients)), '.')
plt.plot(ambients,  orals - physio_correction(surfs_fi_canthus, ambients, 0.125), '.')

# using only cooler ambients
inds=ambients<29.8
f=np.polyfit(ambients[inds], orals[inds]-surfs_fi_canthus[inds], 1)[0]
F=f/(1-f)

plt.ion()
np.set_printoptions(precision=3)
markersize=6

inds=ambients<33
inds=ambients<29.8

effectives=ambients[inds]
ora=orals[inds]
sur=surfs_fi_canthus[inds]
plt.plot(effectives*1.8+32, ora*1.8+32, 'b.', markersize=markersize)
plt.plot(effectives*1.8+32, sur*1.8+32, 'ks', markersize=markersize)
plt.plot(effectives*1.8+32, physio_correction(sur, effectives)*1.8+32, 'r.', markersize=markersize)
mn=np.min(effectives)*1.8+32
mx=np.max(effectives)*1.8+32
x=np.linspace(mn, mx, 20)
po=np.polyfit(effectives*1.8+32, ora*1.8+32, 1)
ps=np.polyfit(effectives*1.8+32, sur*1.8+32, 1)
pb=np.polyfit(effectives*1.8+32, physio_correction(sur, effectives)*1.8+32, 1)
y=np.polyval(po, x)
plt.plot(x, y, 'b--')
y=np.polyval(ps, x)
plt.plot(x, y, 'k--')
y=np.polyval(pb, x)
plt.plot(x, y, 'r--')
plt.xlabel('Ambient (F)', fontsize=12)
plt.ylabel('Measured (F)', fontsize=12)
plt.legend(['Oral Thermometry','Surface Temperature', 'Corrected Body, F=%.3f'%F])


# simple analysis does not allow us to incorporate ambient temps above 86F (blows up the fraction)
inds=ambients<30
# keeps only ~45% of data
F_fi=(orals.reshape((-1,)) - surfs_fi_canthus)/(surfs_fi_canthus-ambients)
F_flir=(orals.reshape((-1,)) - surfs_flir)/(surfs_flir-ambients)
# restricting to ambients below 30C gives us very close to 2021 analyses: 0.164 for FI, 0.155 for Flir
# reanalyzing without the divisional error: 
flir_inds = surfs_flir>33.5

# manual ROI process combined with original data: 
#self.result_data[self.cur] = [ambient, scanner_ambient, int(self.uuid), oral, surf, uncor_surf, flir, sweat, distance, roi_surf, roi_surf_ambcor, roi_surf_ambdistcor]
results=np.array(dsload('manualroi_data.npy'))
units=results[:,-1]
uuids=results[:,2]
orals=results[:,3]
ambients=results[:,0]
orig_prod_surf=results[:,4]
orig_prod_uncor_surf=results[:,5]
surf_flir=results[:,6]
sweat=results[:,7]
distance=results[:,8]
roi_surf_uncor=results[:,9]
roi_surfcor=results[:,11]

F_values_fi_orig = (orals - orig_prod_surf)/(orig_prod_surf-ambients)
F_values_fi = (orals - roi_surfcor)/(roi_surfcor-ambients)
F_values_flir = (orals - surfs_flir)/(surfs_flir-ambients)

# eliminate sweating, temperatures above 90F
inds=(ambients<30)*(sweat<1)*(uuids!=104.0)*(uuids!=110.0)*(units!=52)
flir_inds=(surf_flir>0)*(inds)

# am expecting FLIR to give a smaller F-value since it accepts maximum pixels on face (forehead, partially insulated hairline and shirtcollars, neck)
F_flir = np.nanmean(np.sort(F_values_flir[flir_inds])[1:-1])
# the product should give a slightly smaller value than a manually drawn ROI (again, more receptive to pixels outside the canthus)
F_fi_prod = np.nanmean(np.sort(F_values_fi_orig[inds])[1:-1])
# 0.11763502678102752
F_fi_roi = np.nanmean(np.sort(F_values_fi[inds])[1:-1])
#0.15423630482596418


