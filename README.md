# elevatedfacestudy2021
Noncontact Body Thermometry Study Analyses, Data and Methods Documentation
For preprint hosted at: https://www.medrxiv.org/content/10.1101/2022.01.28.22269746v1

## Contents of sub-folders:
### study_materials
contains IRB protocol, consent, data collection form and description of summer 2021 tent temperature control system

### bias_to_normal
contains surface temperature and adjusted mode output body mode temperature for five NCITs tested with a 4181 IR calibration target set to various setpoints

### extracted_plots
contains plots and data extracted from a few historical publications, showing similar bias-to-normal curves as seen more recently

## Contents of main area:
Various pngs used in medRXiv paper

### Python .py scripts
- examine_stolaf_data.py
- get_stolaf_demographics.py
- merge_stolaf_data.py (documents process of extracting from image data, ancillary data collection (tent temperature control systems), demographics and merging)
- analyze_canthus_march27.py (specific to work on an NCIT paper in 2023, refers to forehead patches extracted from the image datasets acquired for this work)

### numpy .npy datasets
- stolaf_demographics.npy
Contains height, weight, age, gender (1==male, 0==not male)

- stolaf_full_dataset.npy
full_dataset is shaped (28,4), for (subject,tent)
Note: tents C and D were swapped for first 11 datasets 
because C was the second-hottest tent for first 11, then it was the hottest tent
so now in this .npy, in every subject the order goes from coldest tent to hottest tent

columns are:
0 = Air_temperature_snapshot_room
1 = Oral Temperature
2 = Quick Oral Temperature
3 = Surface canthus temperature inside tent at end of 10-minutes
4 = delta time between end of 10 minutes and snapshot
    for using the last measured env temp and last equil image this must be <2 minutes
    np.sum(full_dataset[:,:,4]>120)/112 is 15% of the data
5 = effective air/environmental temperature in last 100 seconds of tent
6 = Surface canthus temperature outside tent during secondary scans
7 = Device #5 surface temperature
8 = Device #2 Body temperature
9 = Device #3 Body temperature
10 = Device #4 Scan Body temperature
11 = air temperature (not adjusted with background, just raw air) over last 100 seconds in tent

- stolaf_filtered_curve_data.npy
curves_X is shaped (28,4,600), for (subject,tent,seconds)
curves contain the linear interpolated (piecewise cubic with endpoints interpolated from the penultimate data)
 these were also filtered with a Savitsky-Golay filter
curves_fsurfs is the canthus temperatures
curves_fenvironment is the environmental effective temperature

model T(t) = T_f - (T_f - T_o)*exp(-t/tau), where tau ~ 1/RC = 95 seconds plus or minus 20%
         1.5 min,        3 min,      4.5 min,     6 min,       7.5 min,     9 min
        1 tau = 37%, 2 tau = 14%, 3 tau = 5%, 4 tau = 1.8%, 5 tau = 0.7%, 6 tau = 0.2%
air 68->78F   = 4F,         1.4F,       0.5F,        0.2F,         0.07F,     0.02F
skin 93->94.8=0.673,       0.25F,      0.09F,        0.04F
underest      = 0.9F,      0.36F,      0.18F,        0.11F



### csv .csv datasets
- oral_temps.csv
- device_temps_tent[A,B,C,D].csv
- tents_snapshot_airtemps.csv

