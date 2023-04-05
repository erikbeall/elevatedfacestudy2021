#!/usr/bin/python3.7

import cv2
import os
import glob
import random
import argparse
import time
import random
import numpy as np
import matplotlib.pylab as plt
plt.ion()
np.set_printoptions(precision=3)


good_image_list = np.load('roianalysis_good_image_list.npy', allow_pickle=True)
good_image_list = good_image_list.tolist()
canthi_surfs = np.load('roianalysis_canthus_surfs.npy')
canthi_surfs[canthi_surfs<32]=np.nan
a1=canthi_surfs[:,:,0]
a2=canthi_surfs[:,:,1]
a3=canthi_surfs[:,:,2]
surfs=np.nanmedian(canthi_surfs, 2)

scan_data=dsload('companion_full_dataset.npy')
scan_indices=[k for k in scan_data.keys() if 'orig_full' not in k]
#{'images': np.array(dataset['images']), 'distcor': distcor(snap_distances), 'bboxes': dataset['bboxes'], 'face_surf_temps': dataset['face_surf_temps']}

full_dataset = scan_data['orig_full_dataset_noswap']

# full_dataset
paper_orals=(full_dataset[:,:,1]-32)/1.8
paper_effectives=full_dataset[:,:,5]
paper_airs=full_dataset[:,:,11]
paper_dists=full_dataset[:,:,12]
paper_snap_ambs=full_dataset[:,:,0]
paper_surfs=full_dataset[:,:,6]
# 18_1 is missing from scan_data and 0/nan in full_data, strip them from here? or index directly by parsing the key

inds=paper_effectives>0
effectives=paper_effectives[inds]
orals=paper_orals[inds]
dists=paper_dists[inds]
surfs=surfs[inds]
offset=orals-surfs
# collapse and match up the canthi_surfs with the contained data, using good_image_list and average those within 1C of median
p=np.polyfit(effectives, offset, 1)
f=-p[0]
F = f/(1-f)

# primary difference between image-based and product output is the use here of last single image instead of a compendium of 9 corrected images for every measurement
# lot more statistics going on in the product than can be performed on the less-complete data saved
a1=a1[inds]
a2=a2[inds]
a3=a3[inds]
t=np.hstack([effectives, effectives, effectives])
o=np.hstack([orals, orals, orals])
s=np.hstack([a1, a2, a3])
of=o-s
ninds=s>0
p=np.polyfit(t[ninds], of[ninds], 1)

# next, load up the crops and do sampling analysis - Exergen is maxima 2x2, rest are random sampled 2x2 vs mean vs mean of top half
data=np.load('roianalysis_crops_forehead.npy', allow_pickle=True).tolist()
# find indices that have crops in them
tmp=np.array([sum([len(d) for d in data[i]]) for i in range(len(data))])
finds=tmp>0
# 99 of 111 have at least one forehead mask

# these are already subsampled down to the same inds as above
effectives=effectives[finds]
orals=orals[finds]
dists=dists[finds]

# NOTE: dependent on the spot size of the various NCITs
# extech: 5-15cm from target, spot size is 1:8 (1" at 8"), thus it measures a spot from 5/8cm to 18/8 cm or 6.25mm to 22.5mm dia spot
# QY-EWQ-02: 3-5cm distance, uses the SGXV02-100-000 Heimann single-pixel, which has a 5.5um filter, https://www.fatritech.com/php/view.php?aid=2058
# FOV is at least 90 degrees (states 112, but the optical figure shows >50% out to around +-45deg -> 90deg)
# but 90deg is very wide, based on lab experiments it is close to 1:1 spot size, thus it measures between 30mm and 50mm dia spot
# exergen: fixed 15mm cup. Maxima over time
# braun is unknown but looks like an unfocused thermopile with a shield, narrower FOV than the QY, appears to be 60deg, or 1:1
# documentation states up to 2" from forehead. The sensor is further recessed 12mm from the cup surface, so up to 63mm distance (2" plus 12mm)
# hence, a spot size of 12 to 63mm
#
# Therefore, we will assume a spot size of 15mm (exergen exact, reasonably best-case in all except QY, likely better than it is able to achieve)
# at 1m distance, pixel extent of m80 and umicore lens is 5mm (circle of 5mm within a 5mm square pixel, or a circle of 1.414 times this or 7mm)
# or a random grouping of 2x2 pixels for all but the exergen, and the maxima 2x2 pixels for the exergen
# thus, step 1 is random sampling of 2x2 pixels out of crop (ignore those out of bounds)
# step 2 is taking largest for exergen, random sampling for others

orig_distcor = lambda distance: np.polyval(np.array([7.48371074e-07, -2.24795911e-03, 0.918]), np.clip(distance, 488, 1400))
dists=orig_distcor(dists)
use_inds=np.array([i for i in range(len(data)) if sum([len(d) for d in data[i]])>0])
# next, for each mask (or multiple masks for a given image)
results=[]
for i in use_inds:
    crops = [d for d in data[i] if len(d)>0]
    # exergen analysis -> 2 tall band maxima, randomly select a 2x2 pixels with a pixel seeded in each vert band from left to right (no need for continuity up/down)
    #  so go from left to right, random pixel selection, then from the available pixels around it, can I draw a 2x2 with a corner at this selected pixel? 
    #  If not, try another pixel (mark attempt as done) and repeat - repeat and if no pixels left, move to next column. At success, append result and move to next column.
    # 2x2 pixel random sampling for all other NCITs
    max_mean=[]
    max_var=[]
    ncit_mean=[]
    ncit_var=[]
    for crop in crops:
        inds = np.where(crop>0)
        x_iters = np.unique(inds[1])
        # first, exclude any x-locations where there are less than 2 connected pixels. assume didn't happen inside the ROI, but it does happen at edges
        x_iters=np.array([x for x in x_iters if len(np.where(inds[1]==x)[0])>1])
        # at this point, its simply select a 2x2 and average them, then store. So use the _next_ as the target
        max_samples=[]
        for x in x_iters[:-1]:
            seed_inds=np.where(inds[1]==x)[0]
            xs_1=inds[1][seed_inds]
            ys_1=inds[0][seed_inds]
            adjseed_inds=np.where(inds[1]==x+1)[0]
            ys_2=inds[0][adjseed_inds]
            xs_2=inds[1][adjseed_inds]
            # at least 2, so pick 2 adjacent in each - remove the end, and assume we're picking the starting lower left corner
            ys_avail=np.intersect1d(ys_1, ys_2)[:-1]
            if len(ys_avail)==0:
                continue
            y_corner=ys_avail[np.random.randint(len(ys_avail))]
            max_samples.append(np.mean(crop[y_corner:y_corner+2,xs_1[0]:xs_2[0]+1]))
        max_mean.append(np.nanmax(max_samples))
        max_var.append(np.nanstd(max_samples))
        samples = [np.nanmean(crop[i-1:i+1, j-1:j+1]) for i,j in zip(inds[0], inds[1]) if np.sum(crop[i-1:i+1, j-1:j+1].flatten()==0)==0]
        # one-time measurement for NCIT: randomly sample but exclude the outliers
        samples=np.sort(samples)[len(samples)//5:-len(samples)//20]
        ncit_mean.append(samples[np.random.randint(len(samples))])
        ncit_var.append(np.std(samples))
    results.append([i, np.nanmean(max_mean), np.nanmean(max_var), np.nanmean(ncit_mean), np.nanmean(ncit_var)])

results=np.array(results)
# apply same distance correction as used in FI
surfs = results[:,1] - dists
offset=orals - surfs
p=np.polyfit(effectives, offset, 1)
f=-p[0]
F = f/(1-f)
# exergen is F=0.20315, NCIT other is 0.210515

markersize=6
plt.figure()
plt.subplot(1,2,1)
# replicate plot from bioarxiv
physio_correction = lambda surf, amb: surf + (surf-amb)*F
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
plt.legend(['Oral Thermometry','Forehead Surface Temperature', 'Corrected Body, F=%.3f'%F])
plt.savefig('F_value_foreheads.png', bbox_inches='tight',pad_inches = 0.1)

