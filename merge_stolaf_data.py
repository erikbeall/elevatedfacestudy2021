#!/usr/bin/env python3

# THIS SCRIPT WAS USED TO EXTRACT DATA FROM DICTIONARY FILES CONTAINING IMAGES
# consent for sharing images was not obtained, thus this script serves to document the
# data extraction process for anonymization and subsequent analyses

import numpy as np
import scipy
import csv
import glob
from scipy import signal
import copy
import matplotlib.pylab as plt
plt.ion()

imload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1')
dsload = lambda filename: np.load(filename, allow_pickle=True, encoding='latin1').tolist()
txtload = lambda filename: np.genfromtxt(filename, delimiter=',')

import datetime
import time
ts2dt = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
dt2ts = lambda timestamp: time.mktime(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
physio_correction = lambda surf, amb: surf + (surf-amb)*0.236
un_physio_correction = lambda core, amb: core - (core-amb)*0.191
physio_correction = lambda surf, amb: surf + (surf-amb)*0.222519
#background = [np.median(np.sort(d[:45,:].flatten())[:35*20]) for d in self.snapshot_data['images']]
#effective = np.mean(background)*0.7 + np.mean(self.snapshot_data['ambients'])*0.3
#print('T_E=%.1f, C=%.1f'%(effective, physio_correction(np.mean(self.snapshot_data['face_surf_temps'][-4:]), effective)))

# tent A in aug4 data was only elevated a little due to controller failure (subjects E1-E4)
dirs=['aug4', 'aug5', 'aug6', 'aug10', 'aug12']
n_subjects=28
loc_unit36=['D', 'D', 'C', 'C', 'C']
txtload = lambda filename: np.genfromtxt(filename, delimiter=',')
linfit = lambda surfs, times: np.polyfit(times - np.min(times), np.log(0.001 + max(surfs) - surfs), 1)

# data will consist of one row per subject, containing the data for each tent
# secondary list = Flir,CEM,Exergen,Seek = 1,2,3,4
description=['T_air_room_before_A', 'T_oral_A', 'T_surf_init_A', 'T_surf_final_A', 'tau_A', 'T_effective_A', 'T_surf_window_A', 'T_surf_SEC1_A', 'T_core_SEC2_A', 'T_core_SEC3_A', 'T_core_SEC4_A', 'T_air_A', 'D_snap']
full_dataset=np.zeros((n_subjects, 4, len(description)))
# curves data contains up to 120 datapoints and times for each datapoint
curves_temps=np.zeros((n_subjects,4,120))
curves_times=np.zeros((n_subjects,4,120))
curves_ambs=np.zeros((n_subjects,4,120))
curves_dists=np.zeros((n_subjects,4,120))
curves_embedded_amb=np.zeros((n_subjects,4,120))
curves_dangling_amb=np.zeros((n_subjects,4,600))
curves_fsurfs=np.zeros((n_subjects,4,600))
curves_fenvironment=np.zeros((n_subjects,4,600))
curves_fambient=np.zeros((n_subjects,4,600))
curves_fradiative=np.zeros((n_subjects,4,600))
curves_fdistances=np.zeros((n_subjects,4,600))
subject_no=0
filenames=np.zeros((n_subjects,4,6))
all_snap_surfs = []
all_snap_times = []
all_uuids = []
for dayno,d in enumerate(dirs):
    mdir='stolaf_'+d+'/'
    keys=None
    group_data = None
    with open(mdir+'group_data.txt', 'r') as fp:
        reader = csv.reader(fp)
        for r in reader:
            if group_data is None:
                keys=r
                group_data={i: [] for i in r}
            else:
                for i in range(len(keys)):
                    group_data[keys[i]].append(r[i])
    print(group_data)
    # secondaries list MUST be entered in the order they were acquired
    secondaries_data = None
    keys=None
    with open(mdir+'secondaries_data.txt', 'r') as fp:
        reader = csv.reader(fp)
        for r in reader:
            if secondaries_data is None:
                keys=r
                secondaries_data={i: [] for i in r}
            else:
                for i in range(len(keys)):
                    if len(r)>0:
                        secondaries_data[keys[i]].append(r[i])
    print(secondaries_data)
    files = glob.glob(mdir+'scan-fi-00*.npy')
    files.sort()
    snapshots = [np.load(f, allow_pickle=True, encoding='latin1').tolist() for f in files]
    # collate the subject data
    units_list=[int(f.replace('_','-').split('-')[3]) for f in files]
    dnum_list=[int(f.replace('_','-').replace('.npy','').split('-')[4]) for f in files]
    print(dnum_list)
    uuid_list=[f['uuid'] for f in snapshots]
    start_times_list=[f['times'][0] for f in snapshots]
    # swap back the position of tent C - hottest tent was switched for Matt's prototype work outside the hottest tent
    if loc_unit36[dayno] == 'D':
        tents={'36': 'D','47': 'C','37': 'B','43': 'A', '45': 'S'}
    else:
        tents={'36': 'C','47': 'D','37': 'B','43': 'A', '45': 'S'}
    tents_list = [tents[str(unit)] for unit in units_list]
    print(tents)
    print(tents_list)
    print(uuid_list)
    #print(times_list)
    # match by order/time, then confirm uuid and tent match
    # for a subject and their equilibration curves for each tent, match the group_data and secondaries data
    # re-sort everything by time, since the secondaries dataset is sorted by time
    # subject_data = {}
    # {'UUID': ['E1', 'E2', 'E3', 'E4'], 'PreOral': ['98.9', '98.2', '98.4', '98.9'], 'OralAQuick'...
    # OralAQuick OralA, Weight, Height, Gender, Age
    # for each UUID, build up their data line
    for uuid in group_data['UUID']:
        print(subject_no)
        all_uuids.append(uuid)
        index = group_data['UUID'].index(uuid)
        uuid_inds = np.where(np.array(uuid_list)==uuid)[0]
        for tent_no,tent in enumerate(['A', 'B', 'C', 'D']):
            tent_inds = np.where(np.array(tents_list)==tent)[0]
            dset_index = np.intersect1d(tent_inds, uuid_inds)
            if len(dset_index)>1:
                print([tents_list[d] for d in dset_index], [uuid_list[d] for d in dset_index])
                print('')
                print('tent and uuid indices do not match in one spot, missing or too much data', dset_index)
                print('')
                time.sleep(1)
                continue
            if len(dset_index)==0:
                print('')
                print('tent and uuid indices do not match due to MISSING data', dset_index)
                print('')
                # fill in the non-image parts of the dataset
                full_dataset[subject_no, tent_no, 1] = float(group_data['Oral'+tent][index])
                full_dataset[subject_no, tent_no, 2] = float(group_data['Oral'+tent+'Quick'][index])
                full_dataset[subject_no, tent_no, 4] = 600
                # manual re-alignment of this one dataset - missing so failed the auto-timestamp based method below
                dataset = dsload('stolaf_aug10/scan-fi-000045_000114.npy')
                snap_surfs=np.array(dataset['face_surf_temps'])
                snap_surfs = [s for s in snap_surfs if abs(s-np.median(snap_surfs))<1]
                snap_surf=np.mean(snap_surfs)
                known_offset = 1.45
                snap_amb=np.median(dataset['ambients']) - known_offset
                snap_distances=np.median(dataset['distances'])
                snap_times=np.mean(dataset['times'])

                sec_tent_inds = np.where(np.array(secondaries_data['Tent'])==tent)[0]
                sec_uuid_inds = np.where(np.array(secondaries_data['UUID'])==uuid)[0]
                sec_ind = np.intersect1d(sec_tent_inds, sec_uuid_inds)[0]
                secondaries=[secondaries_data['Flir'][sec_ind], secondaries_data['CEM'][sec_ind], secondaries_data['Exergen'][sec_ind], secondaries_data['Seek'][sec_ind]]
                secondaries = [float(s) if not s.isalpha() else 0.0 for s in secondaries]

                full_dataset[subject_no, tent_no, 0] = snap_amb
                full_dataset[subject_no, tent_no, 6] = snap_surf
                full_dataset[subject_no, tent_no, 7] = secondaries[0]
                full_dataset[subject_no, tent_no, 8] = secondaries[1]
                full_dataset[subject_no, tent_no, 9] = secondaries[2]
                full_dataset[subject_no, tent_no, 10] = secondaries[3]
                continue
            dset_index = dset_index[0]
            dataset = snapshots[dset_index]
            times=np.array(dataset['times'])
            surfs=np.array(dataset['face_surf_temps'])
            distances=np.array(dataset['distances'])
            ambs = np.array(dataset['env_ht_temp'][1:])

            # remove outliers in surfs
            inds=np.where(times>0)[0]
            surfs=surfs[inds]
            fl=9
            fm=5
            fa=21
            # indices changed when data collection was updated during course of study
            if len(surfs)<5:
                fa=5
                fl=3
                fm=3
            elif len(surfs)<7:
                fl=3
                fm=3
                fa=7
            elif len(surfs)<9:
                fl=7
                fa=9
            elif len(surfs)<21:
                fa=11
            times=times[inds]
            end_time = copy.copy(times[-1])
            ambs=ambs[inds]
            # re-align time grid to completion of experiment, last acquisition is at 10 minutes (here we force it to be for ease of comparison)
            times = times - times[-1] + 599.0
            distances=distances[inds]
            msurf = signal.medfilt(surfs, fm)
            spikes=np.where(np.abs(msurf-surfs)>0.5)[0]
            surfs[spikes]=msurf[spikes]
            fsurfs = signal.savgol_filter(surfs, fl, 1)
            fsurf_grid = np.interp(np.linspace(times[0], 599.0, 600), times, fsurfs)

            # filter distances and ambient temperatures
            fdistances=signal.savgol_filter(distances, fm, 1)
            fdist_grid = np.interp(np.linspace(times[0], 599.0, 600), times, fdistances)
            # pad ambs and times to one in the future
            ambs = ambs.tolist()
            ambs.append(ambs[-1])
            # interp to 15-seconds in future (note, times is already aligned to end of acquisitions above)
            interp_times = copy.copy(times).tolist()
            interp_times.append(interp_times[-1]+15)
            interp_times = np.array(interp_times) - 15
            # Savitsky-Golay 3-wide 1st order
            fambs = signal.savgol_filter(ambs, 3, 1)
            # smooth and extrapolate the secondary air sensor
            # ambient is median of last 6*5 = 30 seconds, so shift backwards by 15 seconds
            fambs_grid = np.interp(np.linspace(times[0], 599.0, 600), interp_times, fambs)
            ambs=ambs[:-1]

            # Background radiance
            # get the median of the lowest 10% of pixels outside the blackbody regions (sides of faces)
            bg = np.array([np.median(np.sort(d[:45,:].flatten())[:360]) for d in dataset['images']])
            fbg = signal.savgol_filter(bg, fm, 1)
            fbg_grid = np.interp(np.linspace(times[0], 599.0, 600), times, fbg)
            # when its heating, higher effective temperature since convective coeff 
            # is higher and thus air dominates again

            # determine the effective background temperature using the polarity of changes
            # thus when polarity is positive, the tent is heating, and we use 40% times background
            # otherwise we use 60% background when its cooling, because the background lags (is higher)
            direction = (fambs_grid[1:] - fambs_grid[:-1])>0
            direction=direction.tolist()
            # append last direction to unify length
            direction.append(direction[-1])
            alphas=np.array([0.4 if d else 0.6 for d in direction])
            # smooth the alpha-blending over ~10 seconds (so 21-wide)
            alphas = signal.savgol_filter(alphas,fa,1)
            alphas=0.7
            effective_ambs = alphas*fbg_grid + (1-alphas)*fambs_grid
            curves_fsurfs[subject_no, tent_no, :] = fsurf_grid
            curves_fenvironment[subject_no, tent_no, :] = effective_ambs
            curves_fdistances[subject_no, tent_no, :] = fdist_grid
            curves_fradiative[subject_no, tent_no, :] = fbg_grid
            curves_fambient[subject_no, tent_no, :] = fambs_grid

            curves_temps[subject_no, tent_no, :len(surfs)] = surfs
            filenames[subject_no, tent_no,0] = units_list[dset_index]
            filenames[subject_no, tent_no,1] = dnum_list[dset_index]
            filenames[subject_no, tent_no,2] = ord(uuid_list[dset_index][0])
            filenames[subject_no, tent_no,3] = int(uuid_list[dset_index][1])
            curves_times[subject_no, tent_no, :len(surfs)] = times
            curves_ambs[subject_no, tent_no, :len(surfs)] = ambs
            curves_dists[subject_no, tent_no, :len(surfs)] = distances
            # one sensor is embedded in the stem of the FIP, the other is separated from the device by 6" in air (3-wire interface)
            curves_embedded_amb[subject_no, tent_no, :len(surfs)] = np.array(dataset['ambients'])
            curves_dangling_amb[subject_no, tent_no, :len(surfs)] = np.array(dataset['env_ht_temp'][1:])
            p=linfit(surfs, times)
            tau = 1/p[0]
            #effective = np.mean(background)*0.7 + amb*0.3

            #equil_end_time = end_times_list[dset_index]
            equil_end_time = end_time
            # get the snapshot data from unit45 outside the tents
            snapshot_inds = np.where(np.array(tents_list)=='S')[0]
            dset_index = np.intersect1d(snapshot_inds, uuid_inds)
            # this is where the temporal order matters, MUST match the ordering found in the secondaries file
            snap_times = [start_times_list[t] for t in dset_index]
            deltas = [abs(s-equil_end_time) for s in snap_times]
            temporal_order = np.where(deltas==np.min(deltas))[0]
            dset_index=dset_index[temporal_order][0]

            print(dayno, uuid, tent, dnum_list[dset_index], len(surfs))

            filenames[subject_no, tent_no,4] = units_list[dset_index]
            filenames[subject_no, tent_no,5] = dnum_list[dset_index]
            dataset = snapshots[dset_index]
            snap_surfs=np.array(dataset['face_surf_temps'])
            # save the first three snapshots and their time-base
            all_snap_surfs.append(snap_surfs[:3])
            all_snap_times.append(dataset['times'][:3])
            # outlier-rejection if up to one value is crazy (seen in several datasets, other two were always fine
            # in each case the outlier was due to the snapshot pic catching the operator's arm or the window or the Seek blackbody)
            snap_surfs = [s for s in snap_surfs if abs(s-np.median(snap_surfs))<1]
            # one snap had 4 measurements - offset in sampling code
            snap_surf=np.mean(snap_surfs)
            # snap_amb is questionable due to embedded sensor's known offset - this offset was obtained from last day's data and will be used
            # to infer approximate actual ambient corresponding to that of the consensus-calibrated ambient probes in each tent
            # (but we have a consensus-calibrated, air-maintained probe on the snapshot scanner)
            known_offset = 1.45
            snap_amb=np.median(dataset['ambients']) - known_offset
            snap_distances=np.median(dataset['distances'])
            snap_times=np.mean(dataset['times'])
            sec_tent_inds = np.where(np.array(secondaries_data['Tent'])==tent)[0]
            sec_uuid_inds = np.where(np.array(secondaries_data['UUID'])==uuid)[0]
            sec_ind = np.intersect1d(sec_tent_inds, sec_uuid_inds)[0]
            secondaries=[secondaries_data['Flir'][sec_ind], secondaries_data['CEM'][sec_ind], secondaries_data['Exergen'][sec_ind], secondaries_data['Seek'][sec_ind]]
            secondaries = [float(s) if not s.isalpha() else 0.0 for s in secondaries]

            full_dataset[subject_no, tent_no, 0] = snap_amb
            full_dataset[subject_no, tent_no, 1] = float(group_data['Oral'+tent][index])
            full_dataset[subject_no, tent_no, 2] = float(group_data['Oral'+tent+'Quick'][index])
            #full_dataset[subject_no, tent_no, 2] = np.mean(fsurfs[:3])
            full_dataset[subject_no, tent_no, 3] = np.mean(fsurfs[-5:])
            # unfortunately there isn't a tau fit due to the unknown level of convection going on during heating
            full_dataset[subject_no, tent_no, 4] = snap_times - end_time
            # assume last 100 seconds is most relevant, in most cases we snapshotted within 30 seconds of end
            # but in some cases, it was minutes later
            # this is based on the synthetic measured time constant of 94sec for heated faces, we are averaging over one time constant
            full_dataset[subject_no, tent_no, 5] = np.mean(effective_ambs[-100:])
            full_dataset[subject_no, tent_no, 6] = snap_surf
            # flir, CEM, exergen, seek
            full_dataset[subject_no, tent_no, 7] = secondaries[0]
            full_dataset[subject_no, tent_no, 8] = secondaries[1]
            full_dataset[subject_no, tent_no, 9] = secondaries[2]
            full_dataset[subject_no, tent_no, 10] = secondaries[3]
            # store the median of the last 100
            full_dataset[subject_no, tent_no, 11] = np.mean(fambs_grid[-100:])
            # snapshot distance
            full_dataset[subject_no, tent_no, 12] = snap_distances

        subject_no = subject_no + 1

# Device #1's product distance correction at time of study (July/August 2021)
distcor = lambda dist: -np.polyval(np.array([7.484e-07, -2.248e-03, 9.18e-01]), np.clip(dist, 488, 1400))
# apply product distance correction
full_dataset[:,:,6]=full_dataset[:,:,6]+distcor(full_dataset[:,:,12].flatten()).reshape((28,4))

# do same for equilibration measurements
# subject 19, tentB was missing distance data - tentA data was most similar to tentB, with ~17mm difference on average
curves_fdistances[18,1,:]=curves_fdistances[18,0,:]
# median of last 100 seconds of data collection
eqdists=np.nanmedian(curves_fdistances,2)
# apply product distance correction
full_dataset[:,:,3]=full_dataset[:,:,3]+distcor(eqdists).reshape((28,4))

# and adjust surface temperatures inside tents corresponding to distance at time of measurement
for i in range(28):
    for j in range(4):
        curves_fsurfs[i,j,:]=curves_fsurfs[i,j,:]+distcor(curves_fdistances[i,j,:])

np.save('filenames_info.npy', filenames)
np.save('all_data_full.npy', full_dataset)
np.save('fcurve_data.npy', [curves_fsurfs, curves_fenvironment, curves_fdistances, curves_fambient, curves_fradiative])
np.save('uuid_list.npy', all_uuids)
# snapshot assessment for different times
np.save('all_snap_surfs.npy', all_snap_surfs)
np.save('all_snap_times.npy', all_snap_times)

# get f in T_S = T_oral -  f*(T_oral - T_A)
# plot T_S vs (T_oral - T_A) - we have T_S in each tent (and outside tents but we'll experience some reduction)
# invert this via F = f/(1-f) to get T_C = T_S + F*(T_S - T_A)
# backwards direction is f = F/(1+F)
# 1. validate T_effective in every case, make cuts to remove any bad ones
# 2. get new f-parameter, challenge is the variability in actual air and background temperature
# 3. model T(t) = T_f - (T_f - T_o)*exp(-t/tau), where tau ~ 1/RC = 95 seconds plus or minus 20%
#          1.5 min,        3 min,      4.5 min,     6 min,       7.5 min,     9 min
#         1 tau = 37%, 2 tau = 14%, 3 tau = 5%, 4 tau = 1.8%, 5 tau = 0.7%, 6 tau = 0.2%
# air 68->78F   = 4F,         1.4F,       0.5F,        0.2F,         0.07F,     0.02F
# skin 93->94.8=0.673,       0.25F,      0.09F,        0.04F
# underest      = 0.9F,      0.36F,      0.18F,        0.11F

# model based on changing T_effective:
# generate a summation of shrinking deltas to arrive at the same as the exponential model for a given tau
# plt.plot(np.arange(600), 93+1.8*(1-np.exp(-np.arange(600)/94)))
# tau up and down are different depending on how much airflow actually hits the face

# these appear to be all due to heat literally hitting the face (airflow), which will throw off our results
# discards=[4_A, 5_B/C, 7D (fine, just very strong effective_env effects, ust have had fan blowing _in their face_)
# 9D (weird dip at start), 12B, 13C, 18B (simply missing), 20D (already equilibrated???)
# look closer =
def gen_T_equilibration(T_effectives, T_surf_initial, T_oral, tau=94.0):
    surfs=[T_surf_initial]
    for T in T_effectives:
        surfs.append(surfs[-1]+(un_physio_correction(T_oral, T)-surfs[-1])/tau)
    return np.array(surfs[:-1])

# f is the slope of the (T_oral - T_ambient) versus T_ambient (where T_ambient is the "effective" environmental temperature)
F=0.236
f=F/(1+F)
#F = f/(1-f)
un_physio_correction = lambda core, amb: core - (core-amb)*f
physio_correction = lambda surf, amb: surf + (surf-amb)*F
# possible update in near future - POST STUDY - 0.222519

get_elevated_air_temp = lambda air_temp=20.0, core_temp=37.0, core_temp_increase=1.0: un_physio_correction(core_temp+core_temp_increase, air_temp)- ((core_temp - un_physio_correction(core_temp+core_temp_increase, air_temp))/F)
physiologic_elevation = lambda oral, hot_air, norm_air: physio_correction(un_physio_correction(oral, hot_air), norm_air)
physiologic_elevated_temp = lambda oral, hot_air, norm_air: physio_correction(un_physio_correction(oral, hot_air), norm_air)

curves_fsurfs, curves_fenvironment, curves_fdistances, curves_fambient, curves_fradiative = np.load('fcurve_data.npy')

filenames=np.load('filenames_info.npy')
full_dataset=np.load('all_data_full.npy')
# swap tents C and D for first 11 datasets (C was the colder tent for first 11, then it was the hottest tent)
fcp=np.copy(full_dataset)
full_dataset[11:,2]=fcp[11:,3]
full_dataset[11:,3]=fcp[11:,2]
full_dataset[full_dataset==0.0]=np.nan
np.save('stolaf_full_dataset.npy', full_dataset)
fcp=np.copy(curves_fsurfs)
curves_fsurfs[11:,2,:] = fcp[11:,3,:]
curves_fsurfs[11:,3,:] = fcp[11:,2,:]
fcp=np.copy(curves_fenvironment)
curves_fenvironment[11:,2,:] = fcp[11:,3,:]
curves_fenvironment[11:,3,:] = fcp[11:,2,:]
fcp=np.copy(curves_fdistances)
curves_fdistances[11:,2,:] = fcp[11:,3,:]
curves_fdistances[11:,3,:] = fcp[11:,2,:]
fcp=np.copy(curves_fambient)
curves_fambient[11:,2,:] = fcp[11:,3,:]
curves_fambient[11:,3,:] = fcp[11:,2,:]
fcp=np.copy(curves_fradiative)
curves_fradiative[11:,2,:] = fcp[11:,3,:]
curves_fradiative[11:,3,:] = fcp[11:,2,:]

np.save('stolaf_filtered_curve_data.npy', [curves_fsurfs, curves_fenvironment, curves_fdistances, curves_fambient, curves_fradiative])

T_oral=(full_dataset[:,:,1].flatten()-32)/1.8
T_pre=(full_dataset[:,:,2].flatten()-32)/1.8
T_simulated = physio_correction(un_physio_correction(T_oral.reshape((28,4)), full_dataset[:,:,5]), full_dataset[:,:,0])
T_simulated[T_simulated<0]=0.0
T_simulated_FIP = curves_fsurfs

plt.clf()
plt.subplot(2,2,1)
airs=full_dataset[:,:,5].flatten()
airs[airs==0]=np.nan
airs=airs[airs>0]
_=plt.hist(airs*1.8+32,12);
plt.xlabel('Air Temperatures (F)')
plt.subplot(2,2,2)
elevs=np.copy(T_simulated)
elevs[elevs==0]=np.nan
elevs=elevs[elevs>0]
_=plt.hist(elevs*1.8+32,12);
plt.xlabel('Elevated Body Temperatures (F)')
plt.savefig('report_dists.png')


