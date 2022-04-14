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


# Basic TRACKING Parameters
params['detector'] = 'fast'
params['enable_tracker'] = True
params['enable_matcher'] = not params['enable_tracker']
params['klt_use_backward'] = False


plt.figure()

# Different KLT configurations
# 1) no pre-building of the pyramids
params['precompute_pyramids'] = False

with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='no_prepyr')


# 2) pre-building of the pyramids, no gradient
params['precompute_pyramids'] = True
params['pyr_with_derivatives'] = False

with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='prepyr_nograd')


# 3) pre-building of the pyramids with gradient
params['precompute_pyramids'] = True
params['pyr_with_derivatives'] = True

with open(PARAMS_MOD, 'w') as f:
    yaml.dump(params, f)

df = run_sequence(RUN_FILE, PARAMS_MOD, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='prepyr_with_grad')

plt.xlabel('# of Keypoints to track')
plt.ylabel('tracking time (s)')
plt.legend()
plt.grid()

plt.show()