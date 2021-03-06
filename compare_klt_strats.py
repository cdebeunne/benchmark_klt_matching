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

os.nice(1)


# load the default parameters, this file won't be modified, to preserve its formatting
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


# Basic TRACKING Parameters
params['debug'] = False
params['detector'] = 'fast'
params['enable_tracker'] = True
params['enable_matcher'] = not params['enable_tracker']
params['klt_use_backward'] = False


plt.figure('Compar with or without pyramid precomputation')

# Different KLT configurations
# 1) no pre-building of the pyramids
params['precompute_pyramids'] = False

df = run_sequence(RUN_FILE, PARAMS_MOD, params, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='no_prepyr')


# 2) pre-building of the pyramids, no gradient
params['precompute_pyramids'] = True
params['pyr_with_derivatives'] = False

df = run_sequence(RUN_FILE, PARAMS_MOD, params, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='prepyr_nograd')


# 3) pre-building of the pyramids with gradient
params['precompute_pyramids'] = True
params['pyr_with_derivatives'] = True

df = run_sequence(RUN_FILE, PARAMS_MOD, params, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='prepyr_with_grad')

plt.xlabel('# of Keypoints to track')
plt.ylabel('tracking time (s)')
plt.legend()
plt.grid()


# 4) Backward pass?
plt.figure('Backward pass: with or without?')

# WITH
params['precompute_pyramids'] = False
params['klt_use_backward'] = True
df = run_sequence(RUN_FILE, PARAMS_MOD, params, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='With bwd')

# WITHOUT
params['precompute_pyramids'] = False
params['klt_use_backward'] = False
df = run_sequence(RUN_FILE, PARAMS_MOD, params, RESULT_FILE)
plt.plot(df['nb_to_tracks'].to_numpy(), df['dt_track'].to_numpy(), '.', label='Withou bwd')

plt.xlabel('# of Keypoints to track')
plt.ylabel('tracking time (s)')
plt.legend()
plt.grid()

plt.show()