import os
import yaml
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import run_sequence

RUN_FILE = './main_detect_grid'
PARAMS_MOD = 'params_mod.yaml'
RESULT_FILE = 'build/result_detect_grid.csv'
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

df = run_sequence(RUN_FILE, PARAMS_MOD, params, RESULT_FILE)
iters = df.index.to_numpy() 
dt_detect_global = df['dt_detect_global'].to_numpy()
dt_detect_grid = df['dt_detect_grid'].to_numpy()
nb_global = df['nb_global'].to_numpy()
nb_grid = df['nb_grid'].to_numpy()

plt.figure()
plt.title('Time taken for KeyPoints detection = f(t)')
plt.plot(iters, dt_detect_global, '.', label='global')
plt.plot(iters, dt_detect_grid, '.', label='grid')
plt.xlabel('Frame #')
plt.ylabel('Detection time (s)')
plt.grid()
plt.legend()

plt.figure()
plt.title('Overhead of grid based detection = f(t)')
plt.plot(iters, dt_detect_grid - dt_detect_global, '.', label='global')
plt.xlabel('Frame #')
plt.ylabel('Detection time (s)')
plt.grid()
plt.legend()

plt.figure()
plt.title('Number of detection in each frame = f(t)')
plt.plot(iters, nb_global, '.', label='global')
plt.plot(iters, nb_grid, '.', label='grid')
plt.xlabel('Frame #')
plt.ylabel('# of detections')
plt.grid()
plt.legend()


plt.figure()
plt.title('Loss of KeyPoints of grid based detection = f(t)')
plt.plot(iters, nb_global - nb_grid, '.', label='global')
plt.xlabel('Frame #')
plt.ylabel('# of detections')
plt.grid()
plt.legend()

plt.show()


plt.show()