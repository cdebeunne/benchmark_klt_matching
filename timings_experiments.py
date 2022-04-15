import os
import yaml
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import run_sequence


RUN_FILE = './main'
PARAMS_MOD = 'params_mod.yaml'
RESULT_FILE = 'build/result_tracking_timings.csv'
STANDARD_COLORS = ['b','g','r','c','m','y','k']


os.nice(1)


# load the default parameters, this file won't be modified, to preserve its formatting
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


# TRACKING
params['detector'] = 'fast'
# params['detector'] = 'orb'
params['enable_tracker'] = True
params['enable_matcher'] = not params['enable_tracker']

with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
iters_tracking = df.index.to_numpy() 
dt_detect_arr = df['dt_detect'].to_numpy()
dt_track_arr = df['dt_track'].to_numpy()
nb_to_tracks_arr = df['nb_to_tracks'].to_numpy()

plt.figure()
plt.title('dt track = f(nb_to_tracks)')
plt.plot(nb_to_tracks_arr, dt_track_arr, '.')
plt.xlabel('# of Keypoints to track')
plt.ylabel('tracking time (s)')
plt.grid()

plt.figure()
plt.title('dt detect and dt track = f(frame #)')
plt.plot(iters_tracking, dt_detect_arr, '.', label='detect')
plt.plot(iters_tracking, dt_track_arr,  '.', label='track')
plt.xlabel('Frame #')
plt.ylabel('time (s)')
plt.legend()
plt.grid()


# experiment with tracking patch size
patch_sizes = [9,13,17,21]
plt.figure()
plt.title('klt_patch_size trial')
for i, ps in enumerate(patch_sizes):
    params['klt_patch_size'] = ps

    with open(PARAMS_MOD, 'w') as f:
        yaml.dump(params, f)
    df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
    iters_tracking = df.index.to_numpy() 
    dt_detect_arr = df['dt_detect'].to_numpy()
    dt_track_arr = df['dt_track'].to_numpy()
    nb_to_tracks_arr = df['nb_to_tracks'].to_numpy()
    plt.plot(nb_to_tracks_arr, dt_track_arr, STANDARD_COLORS[i]+'.', label=str(ps))

plt.xlabel('# features tracked')
plt.ylabel('time (s)')
plt.legend()


params['klt_patch_size'] = 15
# experiment with tracking patch size
pyramid_levels = [1,2,3,4]
colors = ['r', 'r', ]
plt.figure()
plt.title('nlevels_pyramids_klt trial')
for i, pl in enumerate(pyramid_levels):
    params['nlevels_pyramids_klt'] = pl

    with open(PARAMS_MOD, 'w') as f:
        yaml.dump(params, f)
    df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
    iters_tracking = df.index.to_numpy() 
    dt_detect_arr = df['dt_detect'].to_numpy()
    dt_track_arr = df['dt_track'].to_numpy()
    nb_to_tracks_arr = df['nb_to_tracks'].to_numpy()
    plt.plot(nb_to_tracks_arr, dt_track_arr, STANDARD_COLORS[i]+'.', label=str(pl))

plt.xlabel('# features tracked')
plt.ylabel('time (s)')
plt.legend()


plt.show()