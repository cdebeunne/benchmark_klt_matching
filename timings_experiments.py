import os
import yaml
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_FILE = './main'
PARAMS_MOD = 'params_mod.yaml'
RESULT_FILE = 'build/result_tracking_timings.csv'

os.nice(1)

def run_sequence(run_file, param_file_name, result_file):
    # do this stupid chdir twice before main
    # is not in the same place as the python script    
    os.chdir('build')
    subprocess.run([run_file, param_file_name])
    os.chdir('..')
    df = pd.read_csv(result_file)
    return df


# load the default parameters, this file won't be modified, to preserve its formatting
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


# TRACKING
# params['detector'] = 'fast'  # not working right now
params['detector'] = 'orb'
params['enable_tracker'] = True
params['enable_matcher'] = not params['enable_tracker']
# save a modified parameter file to produce tracking results
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
plt.xlabel('tracking time (ms)')
plt.grid()

plt.figure()
plt.title('dt detect and dt track = f(t)')
plt.plot(iters_tracking, dt_detect_arr, label='detect')
plt.plot(iters_tracking, dt_track_arr, label='track')
plt.xlabel('Frame #')
plt.ylabel('time (ms)')
plt.legend()
plt.grid()





plt.show()