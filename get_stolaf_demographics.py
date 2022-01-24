#!/usr/bin/env python3

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

# tent A in aug4 data was not elevated (maybe a degree)
dirs=['aug4', 'aug5', 'aug6', 'aug10', 'aug12']
n_subjects=28
loc_unit36=['D', 'D', 'C', 'C', 'C']
txtload = lambda filename: np.genfromtxt(filename, delimiter=',')
linfit = lambda surfs, times: np.polyfit(times - np.min(times), np.log(0.001 + max(surfs) - surfs), 1)

# data will consist of one row per subject, containing the data for each tent
# secondary list = Flir,CEM,Exergen,Seek = 1,2,3,4
# M=1, F=0
description=['Weight','Height','Age','Gender']
full_dataset=np.zeros((n_subjects, len(description)))
subject_no=0
sessions=np.zeros((28*4,))
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
    print(group_data['UUID'][0])
    sess=ord(group_data['UUID'][0][0]) - ord('E')
    for i in range(len(group_data['UUID'])):
        gender=0
        if group_data['Gender'][i]=='M':
            gender=1
        full_dataset[subject_no, 0] = float(group_data['Weight'][i])
        full_dataset[subject_no, 1] = float(group_data['Height'][i])
        full_dataset[subject_no, 2] = float(group_data['Age'][i])
        full_dataset[subject_no, 3] = gender
        sessions[subject_no*4:subject_no*4+4] = sess
        subject_no = subject_no + 1

np.save('stolaf_demographics.npy', full_dataset)
np.save('stolaf_sessions.npy', sessions)

