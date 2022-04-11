import os
import yaml
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS_MOD = 'params_mod.yaml'


def run_sequence(param_file_name):
    # do this stupid chdir twice before main
    # is not in the same place as the python script    
    os.chdir('build')
    RUN_FILE = './main_sequence'
    subprocess.run([RUN_FILE, param_file_name])
    os.chdir('..')
    df = pd.read_csv('build/results.csv', index_col='iter')
    
    return df.index.to_numpy(), df['nTracks'].to_numpy()


# load the default parameters, this file won't be modified, to preserve its formatting
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


# TRACKING
params['enable_tracker'] = True
params['enable_matcher'] = not params['enable_tracker']
# save a modified parameter file to produce tracking results
with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

its_tracking, tracks_track = run_sequence(PARAMS_MOD)
print('tracking done')


# MATCHING
params['enable_tracker'] = False
params['enable_matcher'] = not params['enable_tracker']
# save a modified parameter file to produce matching results
with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

its_matching, tracks_match = run_sequence(PARAMS_MOD)
print('matching done')


plt.figure()
plt.plot(its_tracking, tracks_track, 'b', label='tracking')
plt.plot(its_matching, tracks_match, 'g', label='matching')
plt.xlabel('# of frame')
plt.ylabel('# of tracks alive')
plt.xlim(0,60)
plt.legend()
plt.grid()


plt.show()