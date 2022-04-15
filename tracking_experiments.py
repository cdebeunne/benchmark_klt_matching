import os
import yaml
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import run_sequence


RUN_FILE = './main_sequence'
PARAMS_MOD = 'params_mod.yaml'
RESULT_FILE = 'build/results.csv'

os.nice(1)


# load the default parameters, this file won't be modified, to preserve its formatting
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


# TRACKING
params['detector'] = 'fast'  # not working right now
# params['detector'] = 'orb'
params['enable_tracker'] = True
params['enable_matcher'] = not params['enable_tracker']
# save a modified parameter file to produce tracking results
with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
iters_tracking, tracks_track = df.index.to_numpy(), df['nTracks'].to_numpy()
print('tracking done')


# MATCHING
params['detector'] = 'orb'
params['enable_tracker'] = False
params['enable_matcher'] = not params['enable_tracker']
# save a modified parameter file to produce matching results
with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
iters_matching, tracks_match = df.index.to_numpy(), df['nTracks'].to_numpy()
print('matching done')


plt.figure()
plt.plot(iters_tracking, tracks_track, 'b', label='tracking')
plt.plot(iters_matching, tracks_match, 'g', label='matching')
plt.xlabel('# of frame')
plt.ylabel('# of tracks alive')
plt.xlim(0,60)
plt.legend()
plt.grid()


plt.show()