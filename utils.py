import os
import yaml
import subprocess
import pandas as pd

def run_sequence(run_file, param_file_name, params, result_file):

    # first overwrite the parameter file that need to be used
    # with the new parameters
    with open(param_file_name, 'w') as f:
        yaml.dump(params, f)

    # do this stupid chdir twice before main
    # is not in the same place as the python script    
    os.chdir('build')
    subprocess.run([run_file, param_file_name])
    os.chdir('..')
    df = pd.read_csv(result_file)
    return df
